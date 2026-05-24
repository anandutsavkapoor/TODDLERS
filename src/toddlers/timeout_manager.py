from .imports import signal, wraps

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