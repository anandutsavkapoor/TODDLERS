from .imports import signal, wraps, os

class TimeoutManager:
    """
    Class to handle timeout functionality for long-running operations.
    
    This class provides a context manager and decorator for adding timeout
    capabilities to any operation, with proper cleanup of signal handlers.
    
    Attributes:
        timeout (int): Timeout duration in seconds
        original_handler: Original SIGALRM handler
    
    Example::

        with TimeoutManager(3600):
            long_running_operation()

        # Or as a decorator
        @TimeoutManager.timeout(3600)
        def long_running_operation():
            pass
    """
    
    def __init__(self, timeout):
        """
        Initialize the TimeoutManager.
        
        Args:
            timeout (int): Timeout duration in seconds
        """
        self.timeout = timeout
        self.original_handler = None
    
    def _timeout_handler(self, signum, frame):
        """Signal handler for timeout."""
        raise TimeoutError(f"Operation exceeded time limit of {self.timeout} seconds")
    
    def __enter__(self):
        """Set up the timeout handler."""
        self.original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.timeout)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the timeout handler."""
        signal.alarm(0)  # Disable the alarm
        signal.signal(signal.SIGALRM, self.original_handler)
    
    @staticmethod
    def timeout(seconds):
        """
        Decorator for adding timeout to functions.
        
        Args:
            seconds (int): Timeout duration in seconds
            
        Returns:
            callable: Decorated function with timeout capability
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with TimeoutManager(seconds):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class LogStallWatchdog:
    """Escalate a long operation when its *log file stops growing*, not after a flat cap.

    A flat wall-clock timeout punishes a slow-but-progressing run (it kills it even while
    it is still advancing) and is slow to react to a genuinely stuck one. This watchdog
    instead polls the operation's log file: as long as the file keeps growing the run is
    left alone, however slow; only when it has produced **no new output for
    ``stall_timeout`` seconds** is it interrupted. ``hard_cap`` is a generous absolute
    backstop (and the sole trigger when there is no log file to watch, e.g. a null
    logger). It raises ``TimeoutError`` -- exactly like :class:`TimeoutManager` -- so any
    existing escalation/retry logic around it is unchanged.

    Example::

        with LogStallWatchdog(self.logger.log_filepath, stall_timeout=2700,
                              hard_cap=SIMULATION_TIMEOUT):
            long_running_operation()
    """

    def __init__(self, log_path, stall_timeout, poll=60, hard_cap=None):
        """
        Args:
            log_path (str|None): file whose growth signals progress; None -> hard_cap only.
            stall_timeout (int): seconds of no log growth that triggers escalation.
            poll (int): how often (s) to check for growth (>=1; SIGALRM is integer-second).
            hard_cap (int|None): absolute wall-clock ceiling, in seconds, as a backstop.
        """
        self.log_path = log_path
        self.stall_timeout = int(stall_timeout)
        self.poll = max(1, int(poll))
        self.hard_cap = int(hard_cap) if hard_cap else None
        self.original_handler = None

    def _size(self):
        try:
            return os.path.getsize(self.log_path) if self.log_path else None
        except OSError:
            return None

    def _tick(self, signum, frame):
        self._elapsed += self.poll
        if self.hard_cap and self._elapsed >= self.hard_cap:
            raise TimeoutError(
                f"Operation exceeded absolute cap of {self.hard_cap} seconds")
        size = self._size()
        if size is None:
            # nothing to watch -> behave as a plain cap (only hard_cap can fire)
            signal.alarm(self.poll)
            return
        if size != self._last_size:
            self._last_size = size
            self._stalled = 0
        else:
            self._stalled += self.poll
            if self._stalled >= self.stall_timeout:
                raise TimeoutError(
                    f"Log '{self.log_path}' made no progress for "
                    f">= {self.stall_timeout} seconds (stalled) -> escalating")
        signal.alarm(self.poll)

    def __enter__(self):
        self._last_size = self._size()
        self._stalled = 0
        self._elapsed = 0
        self.original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._tick)
        signal.alarm(self.poll)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if self.original_handler is not None:
            signal.signal(signal.SIGALRM, self.original_handler)