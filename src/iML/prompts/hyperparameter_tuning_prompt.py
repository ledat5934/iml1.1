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
You are an expert ML engineer. Your task is to generate a Python script that performs hyperparameter tuning using Optuna on the assembled pipeline.

## CONTEXT
- A combined script `full_pipeline.py` is available and defines:
  1. `preprocess_data(...)` to load and preprocess the dataset.
  2. `train_and_evaluate(...)` to train the model and return validation accuracy.

## TUNING SETTINGS
- Number of trials: {n_trials}
- Direction: {direction}
- Sampler: {sampler}
- Pruner: {pruner}
- Timeout (seconds): {timeout}

## HYPERPARAMETERS TO TUNE
Select a small set of the most impactful hyperparameters (2-4) to tune. Avoid tuning trivial parameters to save time and resources.

## TUTORIAL EXAMPLE
Reference example of Optuna usage to guide your script:
```python
import optuna
def objective(trial):
    # Example hyperparameters for a RandomForest
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    # Build, train, and evaluate model
    score = evaluate_model(n_estimators=n_estimators, max_depth=max_depth)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, timeout=600)
best_params = study.best_params
```

## REQUIREMENTS
1. Import necessary modules: `optuna`, `json`, `pickle`, `sys`, and any others needed.
2. Add the output folder to `sys.path` so `full_pipeline.py` can be imported.
3. Create an Optuna `Study` using the given sampler and pruner, with direction `{direction}`.
4. Define `objective(trial)` that:
   - Calls `preprocess_data(...)` to obtain data splits.
   - Uses `trial.suggest_*` methods to select 2-4 key hyperparameters.
   - Calls `train_and_evaluate(...)` to return validation accuracy.
5. Optimize the study with `n_trials={n_trials}` and `timeout={timeout}`.
6. After tuning, save best parameters to `hyperparam_results.json` and the study object to `optuna_study.pkl`.
7. Wrap the main block with `if __name__ == '__main__'`, handle exceptions printing to stderr and exit with `sys.exit(1)`.
8. Return only the complete Python code in a ```python ... ``` block.
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
