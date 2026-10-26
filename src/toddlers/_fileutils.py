"""Temporary file names and the inter-process lock shared by the writers of shared files
(data-dir star tables and grain files, feedback tables, sidecars).

Several workers (processes, and threads of the local parallel runner) write next to each
other in one data dir, so a temporary name carries the pid, the thread and a random part,
and a lock records its owner, so a lock left by a killed process on this host is broken
instead of blocking every later worker for the full timeout.
"""
import contextlib
import json
import os
import secrets
import shutil
import socket
import threading
import time

#: A lock directory without an owner record older than this [s] is taken as abandoned (the
#: owner record is written right after the mkdir).
UNOWNED_LOCK_AGE_S = 60.0


def unique_tmp_path(path):
    """A temporary name next to ``path``, unique per process, thread and call."""
    return f"{path}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp"


@contextlib.contextmanager
def atomic_text_write(path):
    """Open ``path`` for writing text atomically: the content goes to a unique temporary file
    next to it (unique_tmp_path; mode as for open(), umask applied), renamed over ``path``
    when the block ends without an error. On an error the temporary file is removed and a
    previous ``path`` is left as it was, so a reader never sees a partly written file."""
    tmp = unique_tmp_path(path)
    try:
        with open(tmp, 'w') as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class DirLock:
    """Inter-process lock by atomic mkdir (works on NFS and parallel filesystems).

    The lock directory holds ``owner.json`` (host, pid, start time). A waiter breaks the
    lock when its owner is a dead process on this host, or when it has no owner record
    and is older than :data:`UNOWNED_LOCK_AGE_S`; a lock held from another host is never
    broken (its pid cannot be checked) and the waiter raises after ``timeout`` seconds,
    naming the lock directory and its owner.
    """

    def __init__(self, path, timeout=3600.0, poll=0.2, what='lock'):
        self.path, self.timeout, self.poll, self.what = path, timeout, poll, what
        self._stamp = None        # our owner record while we hold the lock
        self._judged = None       # (owner, dir identity) of the lock last judged abandoned

    def _owner(self, path=None):
        try:
            with open(os.path.join(path or self.path, 'owner.json')) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _identity(path):
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_dev, st.st_ino)

    def _abandoned(self):
        ident = self._identity(self.path)
        owner = self._owner()
        if owner is None:
            try:
                dead = time.time() - os.stat(self.path).st_mtime > UNOWNED_LOCK_AGE_S
            except OSError:
                dead = False
        else:
            dead = owner.get('host') == socket.gethostname() and not _pid_alive(int(owner.get('pid', -1)))
        self._judged = (owner, ident) if dead else None
        return dead

    def _break(self):
        """Rename the abandoned lock away (atomic: one waiter wins), check that the moved
        directory is the one judged abandoned (same owner record, same directory), then
        remove it. If another lock took its place in between (another waiter broke the dead
        one and a new holder took it), that live lock is put back, not removed."""
        judged, self._judged = self._judged, None
        grave = f"{self.path}.abandoned.{secrets.token_hex(4)}"
        try:
            os.rename(self.path, grave)
        except OSError:
            return
        if judged is None or (self._owner(grave), self._identity(grave)) != judged:
            try:
                os.rename(grave, self.path)       # not the lock we judged dead: put it back
            except OSError:
                # the path is taken again; the moved lock's holder no longer has it either way
                shutil.rmtree(grave, ignore_errors=True)
                print(f"  Warning: {self.what} {self.path} changed hands while being broken; "
                      f"a live lock was moved away and could not be put back")
            return
        shutil.rmtree(grave, ignore_errors=True)
        print(f"  Removed an abandoned {self.what} {self.path}")

    def __enter__(self):
        t0 = time.monotonic()
        while True:
            try:
                os.mkdir(self.path)
            except FileExistsError:
                if self._abandoned():
                    self._break()
                    continue
                if time.monotonic() - t0 > self.timeout:
                    raise RuntimeError(
                        f"Timed out after {self.timeout:.0f} s waiting for the {self.what} "
                        f"{self.path} (owner {self._owner()}). If no process holds it, remove "
                        f"that directory and retry.")
                time.sleep(self.poll)
                continue
            stamp = {'host': socket.gethostname(), 'pid': os.getpid(), 'time': time.time()}
            with open(os.path.join(self.path, 'owner.json'), 'w') as f:
                json.dump(stamp, f)
            self._stamp = stamp
            return self

    def __exit__(self, *exc):
        """Remove the lock only if it is still ours (its owner.json is our record): a lock
        that was broken and taken by another holder is left to that holder."""
        stamp, self._stamp = self._stamp, None
        if stamp is not None and self._owner() == stamp:
            shutil.rmtree(self.path, ignore_errors=True)
        elif os.path.isdir(self.path):
            print(f"  Warning: {self.what} {self.path} is no longer held by this process "
                  f"(owner {self._owner()}); left in place")
        return False
