import uuid
from datetime import datetime
from pathlib import Path
import logging
import signal
import sys
import time

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

from .core.manager import Manager
from .utils.rich_logging import configure_logging

class PipelineTimeoutError(Exception):
    """Custom exception for pipeline timeout."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    logger.error("Pipeline execution timed out.")
    raise PipelineTimeoutError("Pipeline execution exceeded the time limit.")

def run_automl_pipeline(input_data_folder: str, output_folder: str = None, config_path: str = "configs/default.yaml", checkpoint_mode: str = "full", checkpoint_action: str = "run", single_iteration: str = None):
    """
    Main function to set up the environment and run the entire pipeline.
    
    Args:
        input_data_folder: Path to input data directory
        output_folder: Path to output directory (auto-generated if None)
        config_path: Path to configuration file
        checkpoint_mode: "full" (complete run), "partial" (stop at checkpoint), or "resume" (continue from checkpoint)
        checkpoint_action: When partial mode - where to stop ("guideline", "profiling", "description")
                          When resume mode - where to start ("preprocessing", "modeling", "assembler")
    """
    # 1. Create the output directory if one is not provided
    if output_folder is None:
        project_root = Path(__file__).parent.parent.parent # Points to the project root
        working_dir = project_root / "runs"
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_uuid = uuid.uuid4().hex[:8]
        folder_name = f"run_{current_datetime}_{random_uuid}"
        output_path = working_dir / folder_name
    else:
        output_path = output_folder

    # Ensure output_path is a Path object and the directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load the configuration file
    config = OmegaConf.load(config_path)

    # 3. Configure the logging system FIRST
    # This is critical to ensure all subsequent logs respect the verbosity level
    configure_logging(output_dir=output_dir, verbosity=config.verbosity)
    
    logger.info(f"Project running. Output will be saved to: {output_dir.resolve()}")
    logger.info(f"Loaded configuration from: {config_path}")
    logger.debug(f"Full configuration details: {OmegaConf.to_yaml(config)}")

    # Set up the timeout alarm
    if hasattr(config, 'pipeline_timeout') and config.pipeline_timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(config.pipeline_timeout)
        logger.info(f"Pipeline timeout set to {config.pipeline_timeout} seconds.")

    manager = None
    start_time = time.time()
    try:
        # 4. Initialize the Manager with the prepared settings
        manager = Manager(
            input_data_folder=input_data_folder,
            output_folder=str(output_dir),  # Pass as string for consistency
            config=config,
        )

        # 5. Start the pipeline run based on checkpoint mode
        if checkpoint_mode == "full":
            if single_iteration:
                # Run single iteration with specified approach
                manager.run_pipeline_single_iteration(single_iteration)
            else:
                manager.run_pipeline()
        elif checkpoint_mode == "multi-iteration":
            manager.run_pipeline_multi_iteration()
        elif checkpoint_mode == "partial":
            success = manager.run_pipeline_partial(stop_after=checkpoint_action)
            if success:
                logger.info(f"Pipeline stopped successfully after {checkpoint_action} stage.")
                logger.info("To continue, run with --checkpoint-mode resume --checkpoint-action preprocessing")
            else:
                logger.error("Partial pipeline execution failed.")
        elif checkpoint_mode == "resume":
            success = manager.resume_pipeline_from_checkpoint(start_from=checkpoint_action)
            if not success:
                logger.error("Resume pipeline execution failed.")
        else:
            logger.error(f"Invalid checkpoint_mode: {checkpoint_mode}. Use 'full', 'multi-iteration', 'partial', or 'resume'.")

    except PipelineTimeoutError as e:
        logger.error(f"Pipeline stopped due to timeout: {e}")
        sys.exit(1) # Exit with a non-zero status code to indicate an error
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        # Disable the alarm
        signal.alarm(0)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.brief(f"Total pipeline execution time: {elapsed_time:.2f} seconds.")

        if manager:
            manager.report_token_usage()
            logger.brief(f"output saved in {output_dir}.")
            manager.cleanup()
