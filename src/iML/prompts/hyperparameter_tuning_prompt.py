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

🚨🚨🚨 **CRITICAL OUTPUT FILE REQUIREMENT - READ FIRST** 🚨🚨🚨
🔥 **ABSOLUTELY MANDATORY - NO EXCEPTIONS**: Your script MUST generate `submission_tuned.csv` 🔥

🚫🚫🚫 **FORBIDDEN FILE NAMES - NEVER USE THESE** 🚫🚫🚫
- ❌ `submission.csv` (belongs to assembler phase)
- ❌ `predictions.csv` (generic name)
- ❌ `results.csv` (generic name) 
- ❌ `output.csv` (generic name)
- ✅ ONLY `submission_tuned.csv` is acceptable ✅

🔴🔴🔴 **EXTREMELY IMPORTANT - READ CAREFULLY** 🔴🔴🔴
- The hyperparameter tuning phase is SPECIFICALLY REQUIRED to generate `submission_tuned.csv`
- This file name distinguishes tuned results from the original `submission.csv` created by the assembler phase
- ANY OTHER FILENAME (including `submission.csv`) WILL BE CONSIDERED A CRITICAL FAILURE
- The system will automatically check for the presence of `submission_tuned.csv` and report failure if not found
- This requirement is HARDCODED into the system architecture and CANNOT be bypassed

📂 **FILE LOCATION REQUIREMENTS**:
- Save `submission_tuned.csv` in the same directory where the hyperparameter tuning script is executed
- Use relative path `'submission_tuned.csv'` (not absolute paths)
- The file will be created in current working directory (`os.getcwd()`)
- Print absolute path after creation: `print(f"File saved at: {{os.path.abspath('submission_tuned.csv')}}")`

🔍 **MANDATORY FILE VERIFICATION** 🔍
Your script MUST include these verification steps:
1. Check that NO `submission.csv` file exists in the current directory before creating output
2. After creating `submission_tuned.csv`, verify it exists and has content
3. Print absolute path of created file for confirmation

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
4. **CRITICALLY IMPORTANT**: After optimization, trains a final model with best parameters and creates `submission_tuned.csv`

## TUNING SETTINGS
- Number of trials: {n_trials}
- Direction: {direction}
- Sampler: {sampler}
- Pruner: {pruner}
- Timeout (seconds): {timeout}
{fast_training_instructions}

## FINAL TRAINING STEP REQUIREMENTS
🔥 **ABSOLUTELY MANDATORY**: After optimization, your script MUST include a final training step that:
1. **Extracts the best hyperparameters** found by Optuna: `best_params = study.best_params`
2. **Re-trains a NEW final model** using these best hyperparameters on the FULL training dataset
3. **Uses the newly trained model** (with best hyperparameters) to generate predictions on test data
4. 🚨 **SAVES PREDICTIONS TO `submission_tuned.csv`** 🚨
5. 📈 **PRINT THE BEST SCORE** using `print(f'Best score: {{study.best_value}}')`

⚠️ **CRITICAL**: You must TRAIN a new model with best_params, NOT just generate predictions without training!

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
8. 🚨 **FINAL TRAINING STEP**: After optimization, use best_params to CREATE and TRAIN a new final model, then generate predictions
9. 📈 **PRINT THE BEST SCORE**: After optimization, print the best score using `print(f"Best score: {{study.best_value}}")`
10. Wrap the main block with `if __name__ == '__main__'`, handle exceptions printing to stderr and exit with `sys.exit(1)`
11. Return only the complete Python code in a ```python ... ``` block

## EXAMPLE STRUCTURE
```python
import optuna
import json
import pickle
import sys
# ... other imports from successful code ...

# Extract file_paths from successful code
file_paths = {{{{...}}}}  # Use exact paths from successful code

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
        
        # 📈 📈 📈 PRINT BEST SCORE - DO NOT REMOVE 📈 📈 📈
        print(f"Best score: {{study.best_value}}")
        
        # Save results
        with open('hyperparam_results.json', 'w') as f:
            json.dump(study.best_params, f)
        with open('optuna_study.pkl', 'wb') as f:
            pickle.dump(study, f)
            
        # 🚫 CRITICAL: Check for forbidden submission.csv file first
        import os
        if os.path.exists('submission.csv'):
            print("⚠️ WARNING: Found existing submission.csv file - this belongs to assembler phase")
            print("⚠️ Hyperparameter tuning must create submission_tuned.csv instead")
        
        print("Training final model with best parameters...")
        best_params = study.best_params
        print(f"Best parameters found: {{best_params}}")
        
        # Load and preprocess data exactly like in successful code
        # ... copy preprocessing logic from successful code ...
        
        # Create NEW final model with best hyperparameters and train it
        # final_model = ModelClass(**best_params)  # Configure with best_params
        # final_model.fit(X_train_full, y_train_full)  # Train on FULL data
        
        # Generate predictions using the TRAINED model with best hyperparameters
        # test_predictions = final_model.predict(X_test_processed)
        
        # 🔥🔥🔥 Create submission_tuned.csv using predictions from TRAINED model 🔥🔥🔥
        # submission_df = pd.DataFrame({{{{'id': test_ids, 'target': test_predictions}}}})
        # submission_df.to_csv('submission_tuned.csv', index=False)  # REMEMBER: NOT submission.csv!
        print("🎉 Final predictions saved to submission_tuned.csv using best hyperparameters! 🎉")
        
        # 🔍 VERIFY FILE CREATION - DO NOT REMOVE THIS CHECK
        if os.path.exists('submission_tuned.csv'):
            file_path = os.path.abspath('submission_tuned.csv')
            file_size = os.path.getsize('submission_tuned.csv')
            print(f"✅ CONFIRMED: submission_tuned.csv file created successfully")
            print(f"✅ Location: {{file_path}}")
            print(f"✅ Size: {{file_size}} bytes")
        else:
            print("🚨 ERROR: submission_tuned.csv file was NOT created!", file=sys.stderr)
            sys.exit(1)
            
        # 🚫 FINAL CHECK: Ensure no accidental submission.csv was created
        if os.path.exists('submission.csv') and not os.path.exists('submission_tuned.csv'):
            print("🚨 CRITICAL ERROR: submission.csv found but submission_tuned.csv missing!", file=sys.stderr)
            print("🚨 This indicates the wrong filename was used!", file=sys.stderr)
            sys.exit(1)
        
    except Exception as e:
        print(f"Error: {{{{e}}}}", file=sys.stderr)
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
        fast_training_mode = tuning_config.get('fast_training_mode', False)
        
        # Generate fast training instructions if enabled
        fast_training_instructions = ""
        if fast_training_mode:
            fast_training_instructions = """

## ⚡ FAST TRAINING MODE ENABLED
⚠️ **CRITICAL**: To fit within time constraints, use reduced training for trials:
- **Reduce epochs**: Use 5-15 epochs for objective function (instead of 50-200)
- **Reduce dataset**: Use 20-30% of training data for validation during tuning
- **Early stopping**: Use aggressive early stopping (patience=2-3)
- **Batch size**: Use larger batch sizes (256-512) to speed up training
- **Model size**: Consider smaller model architectures during tuning

⚠️ **IMPORTANT**: Only the FINAL model training (after optimization) should use full data and epochs!

```python
# Example for objective function (reduced training)
def objective(trial):
    # ... hyperparameter suggestions ...
    
    # FAST TRAINING for screening
    model.fit(
        X_train_subset,  # Use subset of data (e.g., first 30%)
        y_train_subset,
        epochs=10,       # Reduced epochs
        batch_size=256,  # Larger batch size
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=2)]  # Aggressive early stopping
    )
    
    return validation_score

# FINAL TRAINING (after optimization) - use full resources
final_model.fit(
    X_train_full,    # Full training data
    y_train_full,    
    epochs=50,       # Full epochs
    batch_size=64,   # Optimal batch size
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5)]
)
```"""
        
        prompt = self.template.format(
            n_trials=n_trials,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            timeout=timeout,
            fast_training_instructions=fast_training_instructions,
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
