from .imports import multiprocessing, logging, os, datetime, uuid, time
from .constants import *
from .utils import add_banner_to_log, format_value

class CloudyLogger:
    """
    Logger class for Cloudy simulations with clean formatting and comprehensive logging capabilities.

    This class manages logging for Cloudy simulations, creating detailed logs with simulation 
    parameters, inputs, outputs, and error messages. It provides clean, well-formatted output 
    with clear visual demarcation between timesteps and different types of information.

    Features:
        - Thread-safe logging with multiprocessing lock
        - Clean visual formatting with optional timestamps
        - Clear timestep demarcation
        - Consistent number formatting
        - Proper handling of physical quantities and units
        - Comprehensive simulation timing information

    Attributes:
        log_filepath (str): Path to the log file.
        logger (logging.Logger): The logger instance.
        log_lock (multiprocessing.Lock): Lock for thread-safe logging.
        start_time (float): Start time of the simulation.
        last_log_time (float): Time of the last log entry.
        last_timestep (float): Last recorded timestep for demarcation.
        file_handler (logging.FileHandler): Handler for file output.
        timestamp_formatter (logging.Formatter): Formatter for timestamped messages.
        clean_formatter (logging.Formatter): Formatter for clean messages without timestamps.
    """

    def __init__(self, base_log_path):
        """
        Initialize the CloudyLogger.

        Args:
            base_log_path (str): Base path for the log file. The actual filename will be
                                generated with a unique identifier and timestamp.
        """
        # Create unique log filename with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        self.base_log_path = base_log_path
        log_dir = os.path.dirname(self.base_log_path)
        log_filename = os.path.basename(self.base_log_path) + ".log"
        log_name, log_ext = os.path.splitext(log_filename)
        self.log_filepath = os.path.join(log_dir, f"{log_name}_cloudy_uid{unique_id}_ts{timestamp}{log_ext}")
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        
        # Initialize thread-safe logging
        self.log_lock = multiprocessing.Lock()
        self.logger = logging.getLogger(f"{__name__}.{os.getpid()}.{unique_id}")
        if self.logger.hasHandlers():
            self.logger.handlers.clear()  # Avoid duplicate handlers
        self.logger.setLevel(logging.DEBUG)

        # Set up handlers with different formatters
        self.file_handler = logging.FileHandler(self.log_filepath)
        self.file_handler.setLevel(logging.DEBUG)

        # Create two formatters - one with timestamp and one without
        self.timestamp_formatter =  logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.clean_formatter = logging.Formatter('%(message)s')

        self.file_handler.setFormatter(self.timestamp_formatter)
        self.logger.addHandler(self.file_handler)

        # Initialize timing variables
        self.start_time = None
        self.last_log_time = 0
        self.last_timestep = None

        # Add initial banner
        add_banner_to_log(self.log_filepath)
        self.info("TODDLERS Cloudy Simulations Started")
        self.flush()

    def format_row(self, label, value, unit):
        """
        Format a row with proper handling of various value types and units.

        Args:
            label (str): Label for the row.
            value (Union[float, tuple, np.ndarray]): Value to format, can be:
                - Simple numeric value
                - Tuple of (value, unit)
                - NumPy array
            unit (str): Default unit for the value (used if value is not a tuple)

        Returns:
            str: Formatted row string with proper alignment of label, value, and unit.
        """
        # Handle tuple case (value, unit)
        if isinstance(value, tuple):
            val = value[0]
            unit = value[1]  # Use unit from tuple
        else:
            val = value

        # Convert numpy array to scalar if needed
        if isinstance(val, np.ndarray):
            val = val.item()

        # Format the numeric value
        if isinstance(val, (int, float, np.number)):
            formatted_value = format_value(val, '.2e') if abs(val) > 1000 or abs(val) < 0.01 else format_value(val, '.4g')
        else:
            formatted_value = str(val)

        # Return formatted string with proper alignment
        return f"| {label:<25} | {formatted_value:>15} | {unit:<25} |"

    def log_visual_break(self, message=None, char='=', length=100):
        """
        Log a visual break with optional message, without timestamp.

        Args:
            message (str, optional): Message to include in the break. Defaults to None.
            char (str, optional): Character to use for the break line. Defaults to '='.
            length (int, optional): Total length of the break line. Defaults to 80.
        """
        self.file_handler.setFormatter(self.clean_formatter)
        if message:
            break_length = (length - len(message) - 2) // 2
            self.logger.info(char * break_length + f" {message} " + char * break_length)
        else:
            self.logger.info(char * length)
        self.file_handler.setFormatter(self.timestamp_formatter)

    def log_simulation_parameters(self, params):
        """
        Log the simulation parameters at the start of the run.

        Args:
            params (dict): Dictionary containing simulation parameters.
        """
        self.log_visual_break("Simulation Parameters")
        
        for key, value in params.items():
            if isinstance(value, float):
                self.info(self.format_row(key, value, ""))
            else:
                self.info(f"| {key:<25} | {str(value):<15} |")
        
        self.log_visual_break()
        self.flush()

    def log_timestep_demarcation(self, t):
        """ 
        Announce the timestep 

        Args:
            t (float): Current simulation time in seconds.
        """
        current_timestep = t / MYR_TO_SEC
        # Only add timestep header if it's a new timestep
        if self.last_timestep is None or current_timestep != self.last_timestep:
            self.log_visual_break(f"\n", length=0)
            self.log_visual_break(f"TIMESTEP: t = {format_value(current_timestep, '.2f')} Myr")
            self.log_visual_break(f"\n", length=0)
            self.last_timestep = current_timestep

    def log_model_start(self, t, model_type):
        """
        Log the start of a specific model simulation.

        Args:
            t (float): Current simulation time in seconds.
            model_type (str): Type of model being simulated ('shell', 'unified', or 'dig').
        """
        current_timestep = t / MYR_TO_SEC
        self.info(f"Starting {model_type} model simulation at t = {format_value(current_timestep, '.2f')} Myr")
        self.log_visual_break(char='-')
        self.flush()

    def log_physical_conditions(self, t, conditions):
        """
        Log physical conditions at a specific time.

        Args:
            t (float): Current simulation time in seconds.
            conditions (dict): Dictionary of physical conditions to log.
                             Keys are condition names, values are (value, unit) tuples.
        """
        self.info(f"Physical conditions at t = {format_value(t/MYR_TO_SEC, '.2f')} Myr:")
        self.log_visual_break(char='-')
        
        for key, value in conditions.items():
            self.debug(self.format_row(key, value, conditions.get(f"{key}_unit", "")))
        
        self.log_visual_break(char='-')
        self.flush()

    def log_shell_properties(self, t, properties):
        """
        Log shell properties at a specific time.

        Args:
            t (float): Current simulation time in seconds.
            properties (dict): Dictionary of shell properties to log.
                             Keys are property names, values are (value, unit) tuples.
        """
        self.info(f"Shell properties at t = {format_value(t/MYR_TO_SEC, '.2f')} Myr:")
        self.log_visual_break(char='-')
        
        for key, value in properties.items():
            self.debug(self.format_row(key, value, properties.get(f"{key}_unit", "")))
        
        self.log_visual_break(char='-')
        self.flush()

    def log_model_end(self, t, model_type, success, duration=None):
        """
        Log the end of a specific model simulation.

        Args:
            t (float): Current simulation time in seconds.
            model_type (str): Type of model being simulated.
            success (bool): Whether the simulation was successful.
            duration (float, optional): Duration of the simulation in seconds.
        """
        current_timestep = t / MYR_TO_SEC
        status = "successfully" if success else "unsuccessfully"
        message = f"{model_type} model simulation at t = {format_value(current_timestep, '.2f')} Myr completed {status}"
        if duration is not None:
            message += f" (Duration: {format_value(duration, '.2f')}s)"
        
        self.info(message)
        self.log_visual_break(char='-')
        self.flush()

    def log_error(self, t, model_type, error_message):
        """
        Log an error that occurred during simulation.

        Args:
            t (float): Current simulation time in seconds.
            model_type (str): Type of model being simulated.
            error_message (str): Error message to log.
        """
        self.error(f"Error in {model_type} model at t = {format_value(t/MYR_TO_SEC, '.2f')} Myr:")
        self.error(error_message)
        self.log_visual_break(char='-')
        self.flush()

    def start_timer(self):
        """Start the simulation timer."""
        self.start_time = time.time()
        self.info("Cloudy simulation timer started.")
        self.flush()

    def log_total_time(self):
        """Log the total simulation time with clear final demarcation."""
        if self.start_time is None:
            self.warning("Cannot calculate total time: start time was not recorded.")
            return

        end_time = time.time()
        total_time = end_time - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        
        self.log_visual_break("SIMULATION COMPLETED")
        self.info(f"Total Cloudy simulation time: {time_str}")
        self.flush()

    def info(self, message):
        """Log an info message."""
        with self.log_lock:
            self.logger.info(message)

    def debug(self, message):
        """Log a debug message."""
        with self.log_lock:
            self.logger.debug(message)

    def warning(self, message):
        """Log a warning message."""
        with self.log_lock:
            self.logger.warning(message)

    def error(self, message):
        """Log an error message."""
        with self.log_lock:
            self.logger.error(message)

    def flush(self):
        """Flush the log buffer."""
        for handler in self.logger.handlers:
            handler.flush()

    def close(self):
        """
        Close the logger, its handlers, and associated file resources.
        """
        try:
            # Flush any remaining content
            self.flush()
            
            # Close and remove handlers
            if hasattr(self, 'logger'):
                for handler in self.logger.handlers[:]: 
                    try:
                        handler.flush()
                        handler.close()
                        self.logger.removeHandler(handler)
                    except Exception as e:
                        print(f"Error closing handler: {str(e)}")
            
            # Close log file if it exists
            if hasattr(self, 'log_file'):
                try:
                    self.log_file.flush()
                    self.log_file.close()
                except Exception as e:
                    print(f"Error closing log file: {str(e)}")
                    
        except Exception as e:
            print(f"Error during logger cleanup: {str(e)}")
        
        finally:
            # Clear references
            if hasattr(self, 'logger'):
                self.logger.handlers.clear()
