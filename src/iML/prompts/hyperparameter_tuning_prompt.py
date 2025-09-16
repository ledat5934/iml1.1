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
You are an expert ML engineer. Your task is to analyze the provided successful ML pipeline code and generate a Python script that performs hyperparameter tuning using Optuna.

## SUCCESSFUL PIPELINE CODE TO ANALYZE
Below is the complete, working pipeline code that successfully trained a model and generated predictions:

```python
{final_executable_code}
```

## YOUR TASK
Analyze the above code and create a hyperparameter tuning script that:
1. **Extracts the exact file paths** used in the successful pipeline 
2. **Reuses the preprocessing logic** from the successful code
3. **Creates a tunable version** of the model training process
4. **Saves tuned results** to a separate submission file (submission_tuned.csv)

## TUNING SETTINGS
- Number of trials: {n_trials}
- Direction: {direction}
- Sampler: {sampler}
- Pruner: {pruner}
- Timeout (seconds): {timeout}

## HYPERPARAMETERS TO TUNE
Select a small set of the most impactful hyperparameters (2-4) to tune. Avoid tuning trivial parameters to save time and resources.

## ANALYSIS INSTRUCTIONS
1. **Extract file_paths**: Look for the `file_paths` variable or file loading logic in the successful code
2. **Identify preprocess function**: Find how data preprocessing is done
3. **Identify train function**: Find the model training and evaluation logic
4. **Identify hyperparameters**: Look for model parameters that can be tuned (learning_rate, n_estimators, etc.)

## REQUIREMENTS
1. Import necessary modules: `optuna`, `json`, `pickle`, `sys`, and any others needed.
2. **CRITICAL**: Extract and use the exact same file paths and data loading logic from the successful code.
3. Recreate the preprocessing steps from the successful code.
4. Create an Optuna `Study` using the given sampler and pruner, with direction `{direction}`.
5. Define `objective(trial)` that:
   - Uses the extracted file paths to load data exactly like the successful code
   - Uses `trial.suggest_*` methods to select 2-4 key hyperparameters.
   - Trains the model with suggested hyperparameters and returns validation accuracy.
6. Optimize the study with `n_trials={n_trials}` and `timeout={timeout}`.
7. After tuning, save best parameters to `hyperparam_results.json` and the study object to `optuna_study.pkl`.
8. **IMPORTANT**: Save the best tuned model predictions to `submission_tuned.csv` (not submission.csv) in the current working directory.
9. Wrap the main block with `if __name__ == '__main__'`, handle exceptions printing to stderr and exit with `sys.exit(1)`.
10. Return only the complete Python code in a ```python ... ``` block.

## EXAMPLE STRUCTURE
```python
import optuna
import json
import pickle
import sys
# ... other imports from successful code ...

# Extract file_paths from successful code
file_paths = {{...}}  # Use exact paths from successful code

def objective(trial):
    # Suggest hyperparameters
    param1 = trial.suggest_int('param1', 10, 100)
    param2 = trial.suggest_float('param2', 0.01, 1.0)
    
    # Preprocess data (copy from successful code)
    # ... preprocessing logic ...
    
    # Train model with suggested hyperparameters
    # ... training logic ...
    
    return validation_score

if __name__ == "__main__":
    try:
        study = optuna.create_study(direction='{direction}')
        study.optimize(objective, n_trials={n_trials}, timeout={timeout})
        
        # Save results
        with open('hyperparam_results.json', 'w') as f:
            json.dump(study.best_params, f)
        with open('optuna_study.pkl', 'wb') as f:
            pickle.dump(study, f)
            
        # Train final model with best params and save to submission_tuned.csv
        # ... final training and prediction logic ...
        # Make sure to save to current working directory as submission_tuned.csv
        
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)
```
"""

    def build(self, tuning_config: Dict[str, Any], final_executable_code: str = "") -> str:
        """
        Build the prompt string using tuning parameters from config and final executable code.
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
            timeout=timeout,
            final_executable_code=final_executable_code
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
