from .imports import traceback, contextmanager, sys, os, tempfile

class ManualEventTermination(Exception):
    """Exception raised for manual termination of simulation events."""
    def __init__(self, message, t, y):
        super().__init__(message)
        self.t = t
        self.y = y
        self.identifier = "MANUAL_TERMINATION"  # Unique identifier

class NegativeEnergyError(Exception):
    """Exception raised when energy or pressure becomes negative during simulation."""
    def __init__(self, message, t, y, P_b=None):
        super().__init__(message)
        self.t = t
        self.y = y
        self.P_b = P_b
        self.identifier = "NEGATIVE_ENERGY_OR_PRESSURE"

@contextmanager
def capture_output():
    """Capture output of stdout and stderr."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    with tempfile.TemporaryFile(mode='w+') as stdout_buf, tempfile.TemporaryFile(mode='w+') as stderr_buf:
        stdout_save = os.dup(stdout_fd)
        stderr_save = os.dup(stderr_fd)

        os.dup2(stdout_buf.fileno(), stdout_fd)
        os.dup2(stderr_buf.fileno(), stderr_fd)

        try:
            yield lambda: (stdout_buf, stderr_buf)
        finally:
            os.dup2(stdout_save, stdout_fd)
            os.dup2(stderr_save, stderr_fd)
            os.close(stdout_save)
            os.close(stderr_save)

