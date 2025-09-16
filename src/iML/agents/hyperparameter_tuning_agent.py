import logging
import os
from typing import Any, Dict
import optuna
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)

class HyperparameterTuningAgent(BaseAgent):
    """
    Agent to perform hyperparameter tuning using Optuna.
    """
    def __init__(self, config: Any, manager: Any, llm_config: Any=None, **kwargs):
        super().__init__(config, manager)
        # number of trials from config or default
        self.n_trials = getattr(config, 'hyperparameter_tuning', {}).get('n_trials', 50)
        # Initialize LLM and prompt handler if llm_config provided
        if llm_config:
            self.llm = init_llm(
                llm_config, agent_name='hyperparameter_tuning', multi_turn=False
            )
            from ..prompts.hyperparameter_tuning_prompt import HyperparameterTuningPrompt
            self.prompt_handler = HyperparameterTuningPrompt(
                manager=manager, llm_config=llm_config
            )

    def __call__(self, iteration_type=None) -> Dict[str, Any]:
        """
        Executes hyperparameter tuning and returns best parameters.
        """
        self.manager.log_agent_start("Starting hyperparameter tuning phase...")

        # The manager.output_folder is already set to the iteration folder during multi-iteration
        # e.g., /output/dog-breed-identification/iteration_2_custom_nn
        final_code_path = os.path.join(self.manager.output_folder, 'states', 'final_executable_code.py')
        
        if not os.path.exists(final_code_path):
            logger.error(f"final_executable_code.py not found at {final_code_path}. Cannot proceed with hyperparameter tuning.")
            return {"status": "failed", "error": f"final_executable_code.py not available at {final_code_path}"}
        
        # Read the successful code to pass to LLM for analysis
        try:
            with open(final_code_path, 'r', encoding='utf-8') as f:
                final_executable_code = f.read()
            logger.info(f"Successfully read final_executable_code.py from {final_code_path}")
        except Exception as e:
            logger.error(f"Failed to read final_executable_code.py: {e}")
            return {"status": "failed", "error": f"Failed to read final code: {e}"}

        # Prepare hyperparameter tuning retries
        tuning_config = getattr(self.manager.config, 'hyperparameter_tuning', {})
        max_retries = tuning_config.get('max_retries', 5)
        timeout = tuning_config.get('timeout', 3600)
        # Initial prompt-based script
        if not hasattr(self, 'prompt_handler'):
            logger.error("Prompt handler not configured for hyperparameter tuning.")
            return {"status": "failed", "error": "No prompt handler available for tuning."}
        
        # Pass the final executable code to LLM for analysis and tuning script generation
        prompt = self.prompt_handler.build(tuning_config, final_executable_code)
        tuning_script = self.prompt_handler.parse(self.llm.assistant_chat(prompt))
        # Prepend working dir and path injection (no import needed as LLM will extract paths from code)
        injection = (
            "import os, sys\n"
            f"os.chdir(r'{self.manager.output_folder}')\n"
            f"sys.path.insert(0, r'{self.manager.output_folder}')\n"
        )
        tuning_script = injection + tuning_script
        # Execute tuning script with retry and repair via LLM on errors
        max_retries = getattr(self.manager.config.hyperparameter_tuning, 'max_retries', 5)
        last_stdout = None
        last_stderr = None
        for attempt in range(1, max_retries + 1):
            # Write and run
            result = self.manager.execute_code(
                code_to_execute=tuning_script,
                phase_name='hyperparameter_tuning',
                attempt=attempt
            )
            if result.get('success'):
                last_stdout = result.get('stdout')
                break
            # On failure, log and save for debugging
            last_stderr = result.get('stderr', '')
            error_to_log = last_stderr.split('\n')[-10:]
            # Prepare error text without backslashes in f-string expression
            error_text = "\n".join(error_to_log)
            self.manager.save_and_log_states(
                f"---ATTEMPT {attempt} FAILED---\nSCRIPT:\n{tuning_script}\n\nERROR:\n{error_text}",
                f"hpt_attempt_{attempt}_failed.log"
            )
            logger.warning(f"Hyperparameter tuning attempt {attempt} failed. Retrying...")
            # Build repair prompt for LLM with original code context
            repair_prompt = (
                f"The hyperparameter tuning script failed on attempt {attempt} with error:\n{last_stderr}\n"
                "Please fix the following Python script for hyperparameter tuning and return the corrected full script only.\n"
                "Original successful code for reference:\n```python\n" + final_executable_code + "\n```\n"
                "Failed tuning script:\n```python\n" + tuning_script + "\n```"
            )
            response = self.llm.assistant_chat(repair_prompt)
            tuning_script = self.prompt_handler.parse(response)
        else:
            logger.error(f"Hyperparameter tuning failed after {max_retries} attempts.")
            return {"status": "failed", "error": last_stderr}
        # Save tuning stdout from successful run
        self.manager.save_and_log_states(
            last_stdout or '', 'hyperparameter_tuning_stdout.txt'
        )
        self.manager.log_agent_end("Completed hyperparameter tuning phase.")
        return {"status": "success", "results": last_stdout}
