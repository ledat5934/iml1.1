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
This script will be combined with the provided preprocessing code.

## CONTEXT
- **Dataset Name**: {dataset_name}
- **Task Description**: {task_desc}
- **File Paths**: {file_paths} (LOAD DATA FROM THESE PATHS)
- **Data File Description**: {data_file_description}
- **Output data format**: {output_data_format}
- **Submission file description**: {submission_file_description}

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
2.  Your code should start with necessary imports for modeling.
3.  Define an appropriate `train_and_predict()` function based on the data handling approach.
4.  The main execution block (`if __name__ == "__main__":`) must:
    a. Call preprocess_data() to get the data.
    b. **Handle the data based on the iteration type (see IMPORTANT DATA HANDLING).**
    c. Call your train_and_predict() function.
    d. Save the final predictions to a submission.csv file.
5.  **Critical Error Handling**: The main execution block MUST be wrapped in a `try...except` block. If ANY exception occurs, the script MUST print the error to stderr and **exit with a non-zero status code** (`sys.exit(1)`).
6.  Follow the modeling guidelines for algorithm choice.
7.  Do not use extensive hyperparameter tuning unless specified. Keep the code efficient.
8.  Limit comments in the code.
9.  The submission file must have the same structure (number of columns) as the sample submission file provided in the dataset, but may have different ID. You have to use the test data to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
10. Your final COMPLETE Python code should have only ONE main function. If there are duplicate main function, remove the duplicates and keep only one main function.
11. Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
12. **REPRODUCIBILITY & VALIDATION**: 
    - Always use random_state=42 for ALL random operations (model initialization, etc.)
    - Use the validation set for model evaluation (no cross-validation needed)
    - Print the validation score in the format: "Validation Score: <score_value>"
    - Use the evaluation metrics specified in the modeling guidelines

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
            data_handling_instruction=data_handling,
            submission_file_description=description.get('submission file description', 'N/A')
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
- Use epoch = 15 with early stopping = 3
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
- Use epoch = 15 with early stopping = 3
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
- **Memory**: Data is already loaded into memory and ready for training

## CODE STRUCTURE EXAMPLE:
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys

def train_and_predict(X_train, y_train, X_val, y_val, X_test):
    # Initialize traditional ML model
    # TODO: Replace below lines with the appropriate modeling code based on guidelines
    model = RandomForestClassifier(random_state=42, n_estimators=100)  # Example model
    
    # Train the model on full training data
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_predictions = model.predict(X_val)
    val_score = accuracy_score(y_val, val_predictions)
    print(f"Validation Score: {{val_score:.4f}}")
    
    # Make predictions on test set
    print("Making predictions on test set...")
    predictions = model.predict(X_test)
    
    return predictions

if __name__ == "__main__":
    try:
        file_paths = {file_paths_main}
        
        # Get preprocessed data (traditional approach returns arrays/DataFrames)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_paths)
        print("Data loaded successfully.")
        
        # Extract test IDs (assuming they're included in preprocessing or index)
        test_ids = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
        
        # Train and predict
        predictions = train_and_predict(X_train, y_train, X_val, y_val, X_test)
        
        # Create submission
        submission_df = pd.DataFrame({{'ID_COLUMN_NAME': test_ids, 'PREDICTION_COLUMN_NAME': predictions}})
        submission_df.to_csv("submission.csv", index=False)
        
        print("Modeling script executed successfully!")
        
    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```"""
        
        elif iteration_type == "custom_nn":
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **a tuple of generators** (e.g., `train_gen, val_gen, test_gen`) to save memory. Your code must handle these generators efficiently for neural network training.

### Batch Training for Neural Networks:
- **For custom NN**: Use the generators in training loops with proper batching
- **Logic**: Iterate through generators, get batches, and train the neural network incrementally
- **Memory Efficiency**: This approach keeps memory usage low by processing data in batches

## CODE STRUCTURE EXAMPLE:
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import sys


class CustomNN(nn.Module):
    # Define your custom neural network architecture here

def train_and_predict(train_gen, val_gen, test_gen):
    # Model will be initialized after seeing first batch
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training loop
    print("Starting neural network training...")
    # Train your custom neural network with appropriate code.
    # Example: Initialize model, train in epochs with batches
    
    # Validation evaluation
    print("Evaluating on validation set...")
    val_predictions = []  # Replace with actual validation predictions
    val_true = []  # Replace with actual validation labels
    # Example: Evaluate model on validation set using val_gen
    
    val_score = accuracy_score(val_true, val_predictions)  # Replace with appropriate scoring metric
    print(f"Validation Score: {{val_score:.4f}}")
    
    # Test predictions
    print("Making predictions on test set...")
    test_predictions = []  # Replace with actual test predictions
    test_ids = []  # Replace with actual test IDs
    # Example: Generate predictions on test set using test_gen
    
    return test_predictions, test_ids

if __name__ == "__main__":
    try:
        file_paths = {file_paths_main}
        
        # Get data generators
        train_gen, val_gen, test_gen = preprocess_data(file_paths)
        print("Data generators initialized successfully.")
        
        # Train and predict
        predictions, test_ids = train_and_predict(train_gen, val_gen, test_gen)
        
        # Create submission
        submission_df = pd.DataFrame({{'ID_COLUMN_NAME': test_ids, 'PREDICTION_COLUMN_NAME': predictions}})
        submission_df.to_csv("submission.csv", index=False)
        
        print("Modeling script executed successfully!")
        
    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```"""
        
        elif iteration_type == "pretrained":
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns **a tuple of generators** (e.g., `train_gen, val_gen, test_gen`) formatted for pretrained models. Your code must handle these generators efficiently.

### Pretrained Model Data Handling (PyTorch preferred):
- **For pretrained models**: Use generators with PyTorch DataLoader and HuggingFace datasets
- **Logic**: Convert generators to PyTorch DataLoader format for efficient batching
- **PyTorch Integration**: Use torch.utils.data.DataLoader, transformers.Trainer, or custom PyTorch training loops
- **HuggingFace**: Prefer transformers library with PyTorch backend over TensorFlow
- **Compatibility**: Ensure data format matches PyTorch tensor requirements and model input specifications

## CODE STRUCTURE EXAMPLE:
```python
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score
import sys

def train_and_predict(train_gen, val_gen, test_gen):
    # Load pretrained model (adjust based on your task type)
    # TODO: Replace "pretrained_model_name" with actual model name based on guideline
    model_name = "pretrained_model_name" 
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add task-specific head
    num_classes = 2  # TODO: Adjust based on your problem
    classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    classifier.to(device)
    
    # TODO: Implement fine-tuning loop
    # Strategy: First freeze the model and fine-tune the classifier, then unfreeze the model and fine-tune both
    # Example:
    # 1. Freeze model parameters: for param in model.parameters(): param.requires_grad = False
    # 2. Train classifier only for few epochs
    # 3. Unfreeze model: for param in model.parameters(): param.requires_grad = True  
    # 4. Fine-tune both model and classifier with lower learning rate
    
    # Validation evaluation
    print("Evaluating on validation set...")
    val_predictions = []  # TODO: Replace with actual validation predictions from model
    val_true = []  # TODO: Replace with actual validation labels from val_gen
    # Example: Iterate through val_gen, get predictions from model+classifier
    
    val_score = accuracy_score(val_true, val_predictions)  # TODO: Use appropriate metric
    print(f"Validation Score: {{val_score:.4f}}")
    
    # Test predictions  
    print("Making predictions on test set...")
    test_predictions = []  # TODO: Replace with actual test predictions from model
    test_ids = []  # TODO: Replace with actual test IDs from test_gen
    # Example: Iterate through test_gen, get predictions from model+classifier
    
    return test_predictions, test_ids

if __name__ == "__main__":
    try:
        file_paths = {file_paths_main}
        
        # Get data generators
        train_gen, val_gen, test_gen = preprocess_data(file_paths)
        print("Data generators initialized successfully.")
        
        # Train and predict
        predictions, test_ids = train_and_predict(train_gen, val_gen, test_gen)
        
        # Create submission
        submission_df = pd.DataFrame({{'ID_COLUMN_NAME': test_ids, 'PREDICTION_COLUMN_NAME': predictions}})
        submission_df.to_csv("submission.csv", index=False)
        
        print("Modeling script executed successfully!")
        
    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```"""
        
        else:
            # Default behavior (fallback)
            return """## IMPORTANT DATA HANDLING
The provided preprocessing code's `preprocess_data` function returns data based on the model type. Handle the data efficiently according to your chosen algorithm.

### General Guidelines:
- **For traditional ML**: Use DataFrames/arrays directly (if provided)
- **For neural networks**: Use generators for memory efficiency (if provided)
- **Validation**: Always use the validation set for model evaluation
- **Consistency**: Ensure data handling matches your algorithm's requirements"""

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
