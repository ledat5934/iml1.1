# src/iML/prompts/hyperparameter_tuning_prompt.py
import json
from typing import Any, Dict
from .base_prompt import BasePrompt

class HyperparameterTuningPrompt(BasePrompt):
    """
    Prompt handler to generate Python code for hyperparameter tuning using Optuna.
    """
    def default_template(self) -> str:
        """
        Default template to request LLM to generate hyperparameter tuning script.
        """
        return """
You are an expert ML engineer. Your task is to generate a complete Python script for hyperparameter tuning using Optuna.

## CONTEXT
- The modeling code file `modeling_code.py` is available and defines:
  - A neural network class `Net`
  - A function `train_and_evaluate(lr, batch_size)` that trains `Net` and returns validation accuracy.

## TUNING PARAMETERS
- Number of trials: {n_trials}
- Optimization direction: {direction}
- Sampler: {sampler}
- Pruner: {pruner}
- Timeout (seconds): {timeout}

## REQUIREMENTS
1. Import necessary modules: `optuna`, `json`, `pickle`, `sys`, and any others needed.
2. Insert the project output folder path to `sys.path` so that `modeling_code.py` can be imported.
3. Create an Optuna study with direction `{direction}`, using sampler `{sampler}` and pruner `{pruner}`.
4. Define an `objective(trial)` that calls `train_and_evaluate` with hyperparameters suggested by the trial:
   - `lr` from `trial.suggest_float('lr', 1e-5, 1e-1, log=True)`
   - `batch_size` from `trial.suggest_categorical('batch_size', [16, 32, 64, 128])`
5. Optimize the study with `n_trials={n_trials}` and `timeout={timeout}`.
6. After optimization, save:
   - Best parameters to `hyperparam_results.json`.
   - The study object to `optuna_study.pkl` using `pickle`.
7. Wrap the entry point in `if __name__ == '__main__'`, catch any exceptions, print errors to stderr, and exit with `sys.exit(1)` on failure.
8. Provide only the Python script enclosed in ```python ... ``` code block.
"""

    def build(self, tuning_config: Dict[str, Any]) -> str:
        """
        Build the prompt string using tuning parameters from config.
        """
        n_trials = tuning_config.get('n_trials', 50)
        direction = tuning_config.get('direction', 'maximize')
        sampler = tuning_config.get('sampler', 'TPESampler()')
        pruner = tuning_config.get('pruner', 'MedianPruner()')
        timeout = tuning_config.get('timeout', 3600)

        prompt = self.template.format(
            n_trials=n_trials,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            timeout=timeout
        )
        self.manager.save_and_log_states(prompt, 'hyperparameter_tuning_prompt.txt')
        return prompt

    def parse(self, response: str) -> str:
        """
        Extract Python code from LLM response.
        """
        if '```python' in response:
            code = response.split('```python')[1].split('```')[0].strip()
        elif '```' in response:
            code = response.split('```')[1].split('```')[0].strip()
        else:
            code = response.strip()
        self.manager.save_and_log_states(code, 'hyperparameter_tuning_script.py')
        return code
