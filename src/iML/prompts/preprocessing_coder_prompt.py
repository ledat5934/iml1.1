# src/iML/prompts/preprocessing_coder_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt

class PreprocessingCoderPrompt(BasePrompt):
    """
    Prompt handler to generate Python code for data preprocessing.
    """

    def default_template(self) -> str:
        """Default template to request LLM to generate code."""
        return """
You are a professional Machine Learning Engineer.
Generate complete and executable Python preprocessing code for the dataset below.
{batch_processing_instruction}
IMPORTANT: DO NOT CREATE DUMMY DATA.
## DATASET INFO:
- Name: {dataset_name}
- Task: {task_desc}
- Input: {input_desc}
- Output: {output_desc}
- Data files: {data_file_desc}
- File paths: {file_paths} (LOAD DATA FROM THESE PATHS)

## PREPROCESSING GUIDELINES:
{preprocessing_guideline}

## TARGET INFO:
{target_info}

## REQUIREMENTS:
1. Generate COMPLETE, EXECUTABLE Python code.
2. Include all necessary imports (pandas, scikit-learn, numpy, etc.).
3. Handle file loading exactly as the provided paths, DO NOT CREATE DUMMY DATA FILES.
4. Follow the preprocessing guidelines exactly.
{data_return_format}
6. Include basic error handling and data validation within the function.
7. Limit comments in the code.
8. Preprocess both the train and test data consistently.
9. IMPORTANT: The main execution block (`if __name__ == "__main__":`) should test the function with the actual file paths.
10. **Critical Error Handling**: The main execution block MUST be wrapped in a `try...except` block. If ANY exception occurs, the script MUST print the error and then **exit with a non-zero status code** using `sys.exit(1)`.
11. DO NOT USE NLTK
12. Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
13. The provided file paths are the only valid paths to load the data. Do not create any dummy data files.

## CODE STRUCTURE:
```python
# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

def preprocess_data(file_paths: dict):
    \"\"\"
    Preprocess data according to guidelines.
    Preprocesses data and returns batch generators.
    Args:
        file_paths: A dictionary of file paths for data splits.
    Returns:
        A tuple of generators, one for each data split (e.g., (train_gen, val_gen, test_gen)).
    \"\"\"
    # Your preprocessing code here
    
    # Placeholder return
    train_generator, val_generator, test_generator = (None, None, None)
    
    return train_generator, val_generator, test_generator

# Test the function
if __name__ == "__main__":
    try:
        # This assumes the script is run from a directory where it can access the paths
        file_paths = {file_paths_main}
        train_gen, val_gen, test_gen = preprocess_data(file_paths)
        print("Generators initialized.")

        
        print("\\nPreprocessing script and generator test executed successfully!")

    except Exception as e:
        print(f"An error occurred during preprocessing test: {{e}}", file=sys.stderr)
        sys.exit(1)
````
"""

    def build(self, guideline: Dict, description: Dict, previous_code: str = None, error_message: str = None, iteration_type: str = None) -> str:
        """Build prompt to generate preprocessing code."""
        
        preprocessing_guideline = guideline.get('preprocessing', {})
        target_info = guideline.get("target_identification", {})

        # Add iteration-specific preprocessing guidance
        iteration_guidance = self._get_iteration_guidance(iteration_type)
        enhanced_guideline = json.dumps(preprocessing_guideline, indent=2)
        if iteration_guidance:
            enhanced_guideline += f"\n\n## ITERATION-SPECIFIC GUIDANCE:\n{iteration_guidance}"

        # Get batch processing instruction and data return format based on iteration
        batch_instruction, data_format = self._get_batch_processing_config(iteration_type)

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            task_desc=description.get('task', 'N/A'),
            input_desc=description.get('input_data', ''),
            output_desc=description.get('output_data', ''),
            data_file_desc=json.dumps(description.get('data file description', {})),
            file_paths=description.get('link to the dataset', []),
            file_paths_main=description.get('link to the dataset', []),
            preprocessing_guideline=enhanced_guideline,
            target_info=json.dumps(target_info, indent=2),
            batch_processing_instruction=batch_instruction,
            data_return_format=data_format
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
2. Generate a new, complete, and corrected version of the Python code that resolves the issue.
3. Ensure the corrected code adheres to all the original requirements.
4. If the error indicates missing modules (ModuleNotFoundError/ImportError), wrap imports in try/except and, in the except block, use subprocess to install the missing package (e.g., `[sys.executable, '-m', 'pip', 'install', '<package>']` with `check=True`), then retry the import.

Generate the corrected Python code:
"""
            prompt += retry_context
        
        self.manager.save_and_log_states(prompt, "preprocessing_coder_prompt.txt")
        return prompt
    
    def _get_iteration_guidance(self, iteration_type: str = None) -> str:
        """Get iteration-specific preprocessing guidance."""
        if iteration_type == "traditional":
            return """
For Traditional ML algorithms (XGBoost, LightGBM, CatBoost):
- Focus on feature engineering for tabular data
- Use categorical encoding (Label/One-hot/Target encoding)
- Apply numerical feature scaling if needed (StandardScaler, MinMaxScaler)
- Handle missing values with appropriate imputation strategies
- Consider feature selection techniques (SelectKBest, RFE)
- Create polynomial or interaction features if beneficial
- Ensure all features are numerical for tree-based models
- Load entire dataset into memory since traditional ML can handle it efficiently
"""
        elif iteration_type == "custom_nn":
            return """
For Custom Neural Networks:
- Apply numerical normalization/standardization (StandardScaler, MinMaxScaler)
- Convert categorical variables to numerical embeddings or one-hot encoding
- Reshape data to proper format for neural network input
- Create train/validation splits suitable for NN training with proper batching
- Apply data augmentation techniques if applicable
- Ensure consistent data types (float32/float64)
- Consider dimensionality reduction if needed
"""
        elif iteration_type == "pretrained":
            return """
For Pretrained Models (preferably PyTorch-based):
- Format data to match pretrained model input requirements
- For text: prepare tokenization compatible with HuggingFace tokenizers (PyTorch format)
- For images: resize and normalize according to model specifications (torchvision transforms)
- For tabular: extract features suitable for pretrained embeddings
- Use PyTorch tensors and data loaders when possible
- Handle special tokens or formatting required by pretrained models
- Ensure data preprocessing matches pretrained model's training format
- Prefer HuggingFace datasets and transformers library (PyTorch backend)
"""
        else:
            return ""
    
    def _get_batch_processing_config(self, iteration_type: str = None) -> tuple[str, str]:
        """Get batch processing instruction and data return format based on iteration type."""
        if iteration_type == "traditional":
            batch_instruction = "IMPORTANT: Load entire dataset into memory for traditional ML algorithms."
            data_format = "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns a tuple of **preprocessed DataFrames/arrays** (e.g., X_train, X_val, X_test, y_train, y_val, y_test)."
        elif iteration_type == "custom_nn":
            batch_instruction = "IMPORTANT: Preprocess data by batch using generators to reduce memory usage for neural network training."
            data_format = "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns a tuple of **generators**, one for each data split (e.g., train_generator, val_generator, test_generator)."
        elif iteration_type == "pretrained":
            batch_instruction = "IMPORTANT: Preprocess data by batch using generators, ensuring compatibility with pretrained model requirements."
            data_format = "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns a tuple of **generators**, one for each data split (e.g., train_generator, val_generator, test_generator)."
        else:
            # Default behavior
            batch_instruction = "IMPORTANT: Preprocess data by batch using generators to reduce memory usage."
            data_format = "5. Create a function `preprocess_data()` that takes a dictionary of file paths and returns a tuple of **generators**, one for each data split (e.g., train_generator, val_generator, test_generator)."
        
        return batch_instruction, data_format

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "preprocessing_code_response.py")
        return code
