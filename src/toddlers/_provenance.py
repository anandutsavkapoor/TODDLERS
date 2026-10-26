"""Code and environment provenance for run records (the evolution log and .dat)."""
import functools
import os
import platform
import subprocess
import sys


@functools.lru_cache(maxsize=1)
def code_provenance():
    """Version, git commit and environment of this TODDLERS installation (computed once per
    process). The git fields are None outside a source checkout of TODDLERS itself."""
    import numpy
    import scipy
    from . import __version__
    pkg = os.path.dirname(os.path.abspath(__file__))
    commit = dirty = None
    try:
        top = subprocess.run(['git', '-C', pkg, 'rev-parse', '--show-toplevel'], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        if top and os.path.realpath(os.path.join(top, 'src', 'toddlers')) == os.path.realpath(pkg):
            commit = subprocess.run(['git', '-C', top, 'rev-parse', 'HEAD'], capture_output=True,
                                    text=True, timeout=5).stdout.strip() or None
            status = subprocess.run(['git', '-C', top, 'status', '--porcelain', '--untracked-files=no'],
                                    capture_output=True, text=True, timeout=5).stdout
            dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        'toddlers': __version__, 'git_commit': commit, 'git_dirty': dirty,
        'python': sys.version.split()[0], 'numpy': numpy.__version__, 'scipy': scipy.__version__,
        'os': platform.platform(),
    }
