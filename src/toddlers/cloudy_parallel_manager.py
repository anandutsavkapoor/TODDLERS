from .imports import os, cpu_count, Lock, dataclass, List, Queue, current_thread, Optional
from .imports import ThreadPoolExecutor, as_completed, psutil, traceback
from .cloudy_simulation_manager import CloudySimulationManager
from .cloudy_output_handler import CloudyOutputHandler
from .constants import *

@dataclass
class TimeStepTask:
    time: float
    model: str
    simulation_id: str
    inner_model_prefix: Optional[str] = None

class ParallelCloudyManager:
    """
    Manages parallel execution of Cloudy simulations with organized logging and user configuration.
    """
    
    def __init__(self, max_workers=None, time_sampling='adaptive', n_points=None,
                 continue_after_dissolution=False, add_dig=True, logU_background=None,
                 small_to_large_ratio=None):
        """
        Initialize the CloudyParallelManager.
        
        Args:
            max_workers (int, optional): Maximum number of worker processes.
            time_sampling (str, optional): Method for time sampling ('adaptive' or 'uniform').
            n_points (int, optional): Number of time points for uniform sampling.
            continue_after_dissolution (bool): Whether to continue after shell dissolution.
            add_dig (bool): Whether to include DIG models.
            logU_background (float, optional): Log of ionization parameter for DIG background.
            small_to_large_ratio (float, optional): Small-to-large grain mass ratio override.
        """
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self._lock = Lock()
        self.task_queue = Queue()
        
        # Store simulation parameters
        self.time_sampling = time_sampling
        self.n_points = n_points
        self.continue_after_dissolution = continue_after_dissolution
        self.add_dig = add_dig
        self.logU_background = logU_background
        self.small_to_large_ratio = small_to_large_ratio
        
        self.running = False

    def _log(self, message, logger, symbol="=", width=80, is_phase=False):
        """
        Unified logging method with structured messages and fallback to print.
        
        Args:
            message (str): Message to log
            logger (Logger): Logger instance, if None prints to console
            symbol (str): Symbol for visual breaks
            width (int): Width of visual breaks
            is_phase (bool): Whether this is a phase transition message
        """
        with self._lock:
            formatted_message = (f"\n{symbol * width}\n{message}\n{symbol * width}\n" 
                               if is_phase else message)
            
            if logger:
                if is_phase:
                    logger.info(formatted_message)
                else:
                    logger.info(message)
            else:
                print(formatted_message)

    def _worker_process(self, task: TimeStepTask, simulation_manager):
        """Execute a single time step task and return success or failure info."""
        thread_name = current_thread().name
        task_time_Myr = task.time/MYR_TO_SEC

        try:
            # Change to the correct directory for the Cloudy simulation
            os.chdir(simulation_manager.cloudy_run_dir)
            
            self._log(
                f"Time: {task_time_Myr:.2f} Myr - Thread: {thread_name}",
                simulation_manager.logger,
                symbol="-",
                width=40
            )
            simulation_manager.logger.info(f"Processing {task.model} model")

            # Generate input file and run the Cloudy simulation
            model_generator = simulation_manager.get_model_generator(task.model)
            if task.model in ['shell', 'unified', 'dissolved']:
                model_generator.write_input_file(task.time)
                simulation_manager.run_simulation(task.time, task.model)

            elif task.model == 'dig':
                if task.time >= simulation_manager.dissolution_time:
                    task.inner_model_prefix = 'shell'
                run_flag = model_generator.write_input_file(
                    task.time,
                    task.inner_model_prefix
                )
                if run_flag:
                    simulation_manager.run_simulation(task.time, task.model)

            output_handler = CloudyOutputHandler(
                task.model,
                task.time,
                absolute_path=simulation_manager.cloudy_run_dir
            )
            if not output_handler.check_cloudy_success():
                raise RuntimeError(
                    f"Simulation did not complete successfully for {task.model} "
                    f"at t={task_time_Myr:.2f} Myr"
                )
            simulation_manager.logger.info(
                f"Simulation for {task.model} at t={task_time_Myr:.2f} Myr "
                "completed successfully."
            )
            return {'success': True}
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            simulation_manager.logger.error(
                f"Error in {thread_name} at t={task_time_Myr:.2f} Myr: "
                f"{error_traceback}"
            )
            # Return failure info instead of appending
            return {
                'success': False,
                'sim_file': task.simulation_id,
                'model': task.model,
                'time_Myr': task_time_Myr,
                'reason': str(e)
            }

    def run_parallel_simulations(self, simulation_files: List[str]):
        """
        Run parallel simulations, logging any failed or unsuccessful runs at the end.
        """
        mem_per_sim = self.estimate_memory_per_simulation()
        failed_runs = []  # Track failures in main thread

        for sim_file in simulation_files:
            simulation_manager = CloudySimulationManager(
                sim_file,
                method=self.time_sampling,
                n_points=self.n_points,
                add_DIG=self.add_dig,
                logU_background=self.logU_background,
                continue_after_dissolution=self.continue_after_dissolution,
                small_to_large_ratio=self.small_to_large_ratio
            )
            
            try:
                time_points = simulation_manager.get_time_points()
                
                self._log(
                    f"Starting simulation: {os.path.basename(sim_file)}", 
                    simulation_manager.logger,
                    symbol="#",
                    is_phase=True
                )
                self._log(
                    f"Number of time points: {len(time_points)}", 
                    simulation_manager.logger
                )
                self._log(
                    f"Number of worker threads: {self.max_workers}", 
                    simulation_manager.logger
                )
                
                # Process models in strict phase order
                for model_phase in ['shell', 'unified', 'dig', 'dissolved']:
                    self._log(
                        f"Starting {model_phase.upper()} Phase",
                        simulation_manager.logger,
                        is_phase=True
                    )
                    
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = []

                        for t in time_points:
                            if psutil.virtual_memory().available < mem_per_sim * 1024 ** 2:
                                simulation_manager.logger.warning(
                                    f"Insufficient memory for t={t/MYR_TO_SEC:.2f} Myr. "
                                    "Skipping."
                                )
                                continue

                            models_to_run, _ = simulation_manager.determine_model_to_run(t)

                            if model_phase in models_to_run:
                                inner_model_prefix = None
                                if model_phase == 'dig':
                                    inner_model_prefix = (
                                        'unified' if simulation_manager.is_within_cloud_interp(t) > 0
                                        else 'shell'
                                    )

                                task = TimeStepTask(
                                    time=t,
                                    model=model_phase,
                                    simulation_id=sim_file,
                                    inner_model_prefix=inner_model_prefix
                                )
                                futures.append(
                                    executor.submit(self._worker_process, task, simulation_manager)
                                )

                        # Process completed futures and collect failures
                        for future in as_completed(futures):
                            if exception := future.exception():
                                simulation_manager.logger.error(f"Exception: {exception}")
                            else:
                                result = future.result()
                                if not result['success']:
                                    failed_runs.append(result)

                    self._log(
                        f"Completed {model_phase.upper()} Phase",
                        simulation_manager.logger,
                        is_phase=True
                    )

                self._log(
                    f"Completed simulation: {os.path.basename(sim_file)}",
                    simulation_manager.logger,
                    symbol="#",
                    is_phase=True
                )

            finally:
                # Log summary of failed runs for this simulation
                if failed_runs:
                    self._log(
                        "Summary of unsuccessful or failed runs:",
                        simulation_manager.logger,
                        symbol="#",
                        is_phase=True
                    )
                    for failure in failed_runs:
                        if failure['sim_file'] == sim_file:  # Only show failures for current sim
                            self._log(
                                f"Run failed: {os.path.basename(sim_file)} - "
                                f"Phase: {failure['model']} - "
                                f"Time: {failure['time_Myr']:.2f} Myr - "
                                f"Reason: {failure['reason']}",
                                simulation_manager.logger
                            )

                if hasattr(simulation_manager, 'logger'):
                    simulation_manager.logger.close()
                    
        # Print final summary to console since loggers are closed
        self._log("All simulations completed", None, symbol="#", is_phase=True)

    @staticmethod
    def estimate_memory_per_simulation():
        """Estimate memory requirements per simulation."""
        return 2.0  # Approximate memory usage in GB

    def get_optimal_concurrent_simulations(self):
        """Calculate optimal number of concurrent simulations based on resources."""
        total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)
        mem_per_sim = self.estimate_memory_per_simulation()
        cpu_limited = self.max_workers
        memory_limited = int(total_memory / mem_per_sim)
        return min(cpu_limited, memory_limited)
