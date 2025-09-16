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

    def __call__(self) -> Dict[str, Any]:
        """
        Executes hyperparameter tuning and returns best parameters.
        """
        self.manager.log_agent_start("Starting hyperparameter tuning phase...")

        # Write the assembled code (preprocessing+model+execution) to a script for tuning
        assembled_code = getattr(self.manager, 'assembled_code', None)
        if not assembled_code:
            logger.error("Assembled code not available for hyperparameter tuning.")
            return {"status": "failed", "error": "Assembled code not available."}
        pipeline_file = os.path.join(self.manager.output_folder, 'full_pipeline.py')
        self.manager.write_code_script(assembled_code, pipeline_file)

        # Build robust hyperparameter tuning script using runpy to import the assembled full_pipeline
        tuning_config = getattr(self.manager.config, 'hyperparameter_tuning', {})
        direction = tuning_config.get('direction', 'maximize')
        sampler = tuning_config.get('sampler', 'None')
        pruner = tuning_config.get('pruner', 'None')
        timeout = tuning_config.get('timeout', 3600)
        tuning_script = f"""
import os, sys, json, optuna, runpy
# Change to output folder where full_pipeline.py resides
os.chdir(r'{self.manager.output_folder}')
sys.path.insert(0, r'{self.manager.output_folder}')
# Load full_pipeline and extract train_and_evaluate function
module = runpy.run_path('full_pipeline.py')
if 'train_and_evaluate' not in module:
    print('train_and_evaluate not found in full_pipeline.py', file=sys.stderr)
    sys.exit(1)
train_and_evaluate = module['train_and_evaluate']
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    return train_and_evaluate(lr=lr, batch_size=batch_size)
if __name__ == '__main__':
    study = optuna.create_study(direction='{direction}', sampler={sampler}, pruner={pruner})
    study.optimize(objective, n_trials={self.n_trials}, timeout={timeout})
    best = study.best_params
    with open('hyperparam_results.json','w') as f:
        json.dump(best,f)
    print(json.dumps(best))
"""
        tuning_file = os.path.join(self.manager.output_folder, 'hyperparameter_tuning.py')
        self.manager.write_code_script(tuning_script, tuning_file)

        # Execute the tuning script content directly
        result = self.manager.execute_code(
            code_to_execute=tuning_script,
            phase_name='hyperparameter_tuning',
            attempt=1
        )
        if not result.get('success'):
            logger.error("Hyperparameter tuning script failed.")
            return {"status": "failed", "error": result.get('stderr')}

        # Save tuning stdout
        self.manager.save_and_log_states(
            result.get('stdout', ''), 'hyperparameter_tuning_stdout.txt'
        )
        self.manager.log_agent_end("Completed hyperparameter tuning phase.")
        return {"status": "success", "results": result.get('stdout')}
