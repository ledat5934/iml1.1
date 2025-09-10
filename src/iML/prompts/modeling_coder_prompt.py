# src/iML/prompts/modeling_coder_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt

class ModelingCoderPrompt(BasePrompt):
    """
    Prompt handler to generate Python code for modeling.
    """

    def default_template(self) -> str:
        """Default template to request LLM to generate modeling code."""
        return """
You are an expert ML engineer. Your task is to generate a COMPLETE and EXECUTABLE Python script for modeling.
This script will be combined with the provided preprocessing code that uses generators.

## CONTEXT
- **Dataset Name**: {dataset_name}
- **Task Description**: {task_desc}
- **File Paths**: {file_paths} (LOAD DATA FROM THESE PATHS)
- **Data File Description**: {data_file_description}
- **Output data format**: {output_data_format}

{data_handling_instruction}
## MODELING GUIDELINES:
{modeling_guideline}


## PREPROCESSING CODE (Do NOT include this in your response):
The following preprocessing code, including a function `preprocess_data(file_paths: dict)`, will be available in the execution environment. You must call it to get the data.
```python
{preprocessing_code}
```

## REQUIREMENTS:
1.  **Generate COMPLETE Python code for the modeling part ONLY.** Do NOT repeat the preprocessing code.
2.  Your code should start with necessary imports for modeling (e.g., `import pandas as pd`, `from sklearn.ensemble import RandomForestClassifier`).
3.  Define a function `train_and_predict(X_train, y_train, X_test)`.
4.  Keep the data loading code of the preprocessing code.
5.  The main execution block (`if __name__ == "__main__":`) must:
    a. Call preprocess_data() to get the data generators.
    b. **Handle the generators based on the chosen model's capability (see IMPORTANT DATA HANDLING). Prefer incremental training (Path 1). Only aggregate the full dataset into memory (Path 2) if the model does not support batch-based training.**
    c. Call your train_and_predict() function.
    d. Save the final predictions to a submission.csv file.
6.  **Critical Error Handling**: The main execution block MUST be wrapped in a `try...except` block. If ANY exception occurs, the script MUST print the error to stderr and **exit with a non-zero status code** (`sys.exit(1)`).
7.  Follow the modeling guidelines for algorithm choice.
8.  Do not use extensive hyperparameter tuning unless specified. Keep the code efficient.
9.  Limit comments in the code.
10. The submission file must have the same structure (number of columns) as the sample submission file provided in the dataset, but may have different ID. You have to use the test data to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
11. Your final COMPLETE Python code should have only ONE main function. If there are duplicate main function, remove the duplicates and keep only one main function.
12. Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.


## CODE STRUCTURE EXAMPLE:
```python
# Your modeling code starts here
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier # Example model that supports partial_fit
import sys
import os

# The 'preprocess_data' function is assumed to be defined here from the context.
# {preprocessing_code}

# The train_and_predict function now needs to handle the training loop
def train_and_predict(train_gen, test_gen):
    # Initialize a model that supports incremental learning
    model = SGDClassifier(random_state=42)

    # 1. Incremental training using the generator
    print("Starting incremental training...")
    # Note: For the first call to partial_fit, you might need to specify all possible class labels
    # We can get them by iterating through the generator once if not known beforehand.
    # For simplicity, we assume they are known or handled within the loop.
    # A robust implementation would collect all unique `y` values first.
    all_classes = np.array([0, 1]) # IMPORTANT: This is a placeholder, must be replaced with actual classes
    for X_batch, y_batch in train_gen:
        model.partial_fit(X_batch, y_batch, classes=all_classes)
    print("Incremental training complete.")

    # 2. Aggregate test data for prediction
    # Prediction is often done on the full test set at once.
    print("Aggregating test data for prediction...")
    X_test_list, test_ids_list = zip(*[batch for batch in test_gen])
    X_test = pd.concat(X_test_list) if isinstance(X_test_list[0], pd.DataFrame) else np.vstack(X_test_list)
    test_ids = np.concatenate(test_ids_list)

    # 3. Make predictions
    predictions = model.predict(X_test)
    
    return predictions, test_ids


if __name__ == "__main__":
    try:
        # These file paths will be available in the execution environment
        file_paths = {file_paths_main}
        
        # 1. Get data generators
        # The number of returned elements must match the preprocess_data function
        train_gen, val_gen, test_gen = preprocess_data(file_paths)
        print("Data generators initialized successfully.")
        
        # 2. Train model and get predictions using the generator-aware function
        predictions, test_ids = train_and_predict(train_gen, test_gen)
        print("Model training and prediction complete.")

        # 3. Create submission file
        submission_df = pd.DataFrame({{'ID_COLUMN_NAME': test_ids, 'PREDICTION_COLUMN_NAME': predictions}})
        submission_df.to_csv("submission.csv", index=False)

        print("Modeling script executed successfully and submission.csv created!")

    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```
"""

    def build(self, guideline: Dict, description: Dict, preprocessing_code: str, previous_code: str = None, error_message: str = None, iteration_type: str = None) -> str:
        """Build prompt to generate modeling code."""
        
        modeling_guideline = guideline.get('modeling', {})
        
        # Add iteration-specific modeling guidance
        iteration_guidance = self._get_iteration_guidance(iteration_type)
        enhanced_guideline = json.dumps(modeling_guideline, indent=2)
        if iteration_guidance:
            enhanced_guideline += f"\n\n## ITERATION-SPECIFIC GUIDANCE:\n{iteration_guidance}"

        # Get data handling instruction based on iteration type
        data_handling = self._get_data_handling_instruction(iteration_type)

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            task_desc=description.get('task', 'N/A'),
            file_paths=description.get('link to the dataset', []),
            file_paths_main=description.get('link to the dataset', []),
            data_file_description=description.get('data file description', 'N/A'),
            output_data_format=description.get('output_data', 'N/A'),
            modeling_guideline=enhanced_guideline,
            preprocessing_code=preprocessing_code,
            data_handling_instruction=data_handling
        )

        if previous_code and error_message:
            retry_context = f"""
## PREVIOUS ATTEMPT FAILED:
The previously generated code failed with an error.

### Previous Code:
```python
{previous_code}
```

### Error Message:
```
{error_message}
```

## FIX INSTRUCTIONS:
1. Analyze the error message and the previous code carefully.
2. Fix the specific issue that caused the error.
3. Ensure your code correctly uses the data returned by the `preprocess_data` function.
4. Generate a new, complete, and corrected version of the Python code that resolves the issue.
5. Adhere to all original requirements.

Generate the corrected Python code:
"""
            prompt += retry_context
        
        self.manager.save_and_log_states(prompt, "modeling_coder_prompt.txt")
        return prompt
    
    def _get_iteration_guidance(self, iteration_type: str = None) -> str:
        """Get iteration-specific modeling guidance."""
        if iteration_type == "traditional":
            return """
For Traditional ML algorithms (XGBoost, LightGBM, CatBoost):
- Focus on feature importance analysis
- Use early stopping for gradient boosting methods
- Optimize for tabular data characteristics
- Use optuna library for hyperparameter tuning
"""
        elif iteration_type == "custom_nn":
            return """
For Custom Neural Networks:
- Design NN architecture from scratch using PyTorch.
- Include proper layer definitions (Dense, Dropout, BatchNormalization)
- Implement training loop with validation monitoring
- Use appropriate loss functions and optimizers (Adam, SGD)
- Add learning rate scheduling and early stopping
- Include model checkpointing for best weights
- Handle overfitting with regularization techniques
- Monitor training/validation loss curves
"""
        elif iteration_type == "pretrained":
            return """
For Pretrained Models (prioritize PyTorch):
- Load and fine-tune pretrained models (PyTorch backend preferred)
- Use transformers library with PyTorch backend for text models
- Use torchvision for vision models (ResNet, ViT, etc.)
- Implement transfer learning approach with proper layer freezing/unfreezing
- Use HuggingFace tokenizers and PyTorch data loaders
- Fine-tune with appropriate learning rates (often lower than training from scratch)
- Handle model adaptation for target task (classification head modification)
- Use PyTorch-specific optimizers and schedulers
- Implement gradual unfreezing strategy if needed
- Prefer torch.nn.functional and PyTorch ecosystem
"""
        else:
            return ""
    
    def _get_data_handling_instruction(self, iteration_type: str = None) -> str:
        """Get data handling instruction based on iteration type."""
        if iteration_type == "traditional":
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **preprocessed DataFrames/arrays** (e.g., X_train, X_val, X_test, y_train, y_val, y_test) that are already loaded into memory. Your code should directly use these preprocessed datasets.

### Data Usage:
- **For traditional ML**: Use the preprocessed DataFrames/arrays directly with scikit-learn models
- **Logic**: Call `X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_paths)` and use them directly
- **Memory**: Data is already loaded into memory and ready for training"""
        
        elif iteration_type == "custom_nn":
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **a tuple of generators** (e.g., `train_gen, val_gen, test_gen`) to save memory. Your code must handle these generators efficiently for neural network training.

### Batch Training for Neural Networks:
- **For custom NN**: Use the generators in training loops with proper batching
- **Logic**: Iterate through generators, get batches, and train the neural network incrementally
- **Memory Efficiency**: This approach keeps memory usage low by processing data in batches"""
        
        elif iteration_type == "pretrained":
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **a tuple of generators** (e.g., `train_gen, val_gen, test_gen`) formatted for pretrained models. Your code must handle these generators efficiently.

### Pretrained Model Data Handling (PyTorch preferred):
- **For pretrained models**: Use generators with PyTorch DataLoader and HuggingFace datasets
- **Logic**: Convert generators to PyTorch DataLoader format for efficient batching
- **PyTorch Integration**: Use torch.utils.data.DataLoader, transformers.Trainer, or custom PyTorch training loops
- **HuggingFace**: Prefer transformers library with PyTorch backend over TensorFlow
- **Compatibility**: Ensure data format matches PyTorch tensor requirements and model input specifications"""
        
        else:
            # Default behavior (generators)
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **a tuple of generators** (e.g., `train_gen, val_gen, test_gen`) to save memory. Your code must handle these generators efficiently. There are two primary ways to do this, depending on the model's capabilities:

### Path 1: Incremental/Batch Training (PREFERRED METHOD)
This is the most memory-efficient approach and should be your default choice.
- **For scikit-learn**: Use models that support the `.partial_fit()` method (e.g., `SGDClassifier`, `MultinomialNB`, `PassiveAggressiveClassifier`).
- **Logic**: You will iterate through the training generator, and for each batch of data, you will call `model.partial_fit(X_batch, y_batch)`.
- **This method AVOIDS loading the entire dataset into memory.**

### Path 2: Full Data Aggregation (FALLBACK METHOD)
Use this method **ONLY IF** the model specified in the `MODELING_GUIDELINES` does **NOT** support incremental training (e.g., `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`).
- **Logic**: Iterate through the generators to collect all batches and aggregate them into a single large dataset in memory (e.g., using `pandas.concat` or `numpy.vstack`).
- **Warning**: Acknowledge that this approach negates the memory-saving benefits of using generators and should only be a fallback."""

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "modeling_code_response.py")
        return code
