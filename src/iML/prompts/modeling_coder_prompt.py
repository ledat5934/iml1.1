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
You are an expert ML engineer. Your task is to generate Python code for modeling, which will be combined with the provided preprocessing code.

## CONTEXT
- **Dataset Name**: {dataset_name}
- **Task Description**: {task_desc}
- **File Paths**: {file_paths} (LOAD DATA FROM THESE PATHS)
- **Data File Description**: {data_file_description}
- **Output data format**: {output_data_format}

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
    a. Call the existing `preprocess_data()` function to get the datasets.
    b. Call your `train_and_predict()` function.
    c. Save the predictions to a `submission.csv` file. The format should typically be two columns: an identifier column and the prediction column.
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
from sklearn.ensemble import RandomForestClassifier
import sys
import os

{preprocessing_code}

def train_and_predict(X_train, y_train, X_test):
    # Your modeling, training, and prediction logic here
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

if __name__ == "__main__":
    try:
        # These file paths will be available in the execution environment
        file_paths = {file_paths_main}
        
        # 1. Preprocess data using the provided function
        # The number of returned elements must match the preprocess_data function
        X_train, X_test, y_train, y_test, test_ids = preprocess_data(file_paths)

        # 2. Train model and get predictions
        predictions = train_and_predict(X_train, y_train, X_test)

        # 3. Create submission file
        submission_df = pd.DataFrame({{'test_id_column_name': test_ids, 'prediction_column_name': predictions}})
        submission_df.to_csv("submission.csv", index=False)

        print("Modeling script executed successfully and submission.csv created!")

    except Exception as e:
        print(f"An error occurred during modeling: {{e}}", file=sys.stderr)
        sys.exit(1)
```
"""

    def build(self, guideline: Dict, description: Dict, preprocessing_code: str, previous_code: str = None, error_message: str = None) -> str:
        """Build prompt to generate modeling code."""
        
        modeling_guideline = guideline.get('modeling', {})

        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            task_desc=description.get('task', 'N/A'),
            file_paths=description.get('link to the dataset', []),
            file_paths_main=description.get('link to the dataset', []),
            data_file_description=description.get('data file description', 'N/A'),
            output_data_format=description.get('output_data', 'N/A'),
            modeling_guideline=json.dumps(modeling_guideline, indent=2),
            preprocessing_code=preprocessing_code
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
