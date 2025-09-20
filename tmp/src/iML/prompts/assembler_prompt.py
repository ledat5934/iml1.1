# src/iML/prompts/assembler_prompt.py
import json
from typing import Dict, Any

from .base_prompt import BasePrompt

class AssemblerPrompt(BasePrompt):
    """
    Prompt handler to assemble and fix final code.
    """

    def default_template(self) -> str:
        """Default template to request LLM to rewrite and fix code."""
        return """
You are a senior ML engineer finalizing a project. You have been given a Python script that combines preprocessing and modeling.
Your task is to ensure the script is clean, robust, and correct.

## CONTEXT


## REQUIREMENTS:
1.  **Final Script**: The output must be a single, standalone, executable Python file and it should be run on the real data.
2.  **Validation Score**: If validation data is available, you MUST calculate and print a relevant validation score.
3.  **Absolute Output Path**: The script MUST save `submission.csv` to the following absolute path: `{output_path}`.
4.  **Error Handling**: Maintain the `try...except` block for robust execution.
5.  **Clarity**: Ensure the final script is clean and well-structured.
6.  **Sample Submission File**: Sample submission file given is for template reference (Columns) only. You have to use the test data or test file to generate predictions and your right submission file. In some cases, you must browse the test image folder to get the IDs and data.
7.  **Do not add any other code.**
8.  **Data Loading**: Keep the data loading code of the preprocessing code. DO NOT CHANGE THE FILE PATHS FROM THE ORIGINAL CODE.



## ORIGINAL CODE:
```python
{original_code}
```
{retry_context}
## INSTRUCTIONS:
Based on the context above, generate the complete and corrected Python code. The output should be ONLY the final Python code.

## FINAL, CORRECTED CODE:
"""

    def build(self, original_code: str, output_path: str, description: Dict, error_message: str = None, iteration_type: str = None) -> str:
        """Build prompt to assemble or fix code."""
        
        retry_context = ""
        if error_message:
            retry_context = f"""
## PREVIOUS ATTEMPT FAILED:
The code above failed with the following error.

### Error Message:
```
{error_message}
```

### FIX INSTRUCTIONS:
1.  Analyze the error message and the original code carefully.
2.  Fix the specific issue that caused the error.
3.  Generate a new, complete, and corrected version of the Python code that resolves the issue and meets all requirements.
4.  If the error indicates a missing module (ModuleNotFoundError/ImportError), modify the final script to wrap critical imports in try/except and, in the except, call subprocess to install the missing package (e.g., `[sys.executable, '-m', 'pip', 'install', '<package>']` with `check=True`), then attempt the import again before proceeding.
"""
        
        # Add iteration-specific assembly guidance
        iteration_guidance = self._get_iteration_guidance(iteration_type)
        if iteration_guidance:
            additional_context = f"\n\n## ITERATION-SPECIFIC CONTEXT:\n{iteration_guidance}"
            retry_context += additional_context
        
        prompt = self.template.format(
            dataset_name=description.get('name', 'N/A'),
            file_paths=description.get('link to the dataset', []),
            output_data_format=description.get('output_data', 'N/A'),
            original_code=original_code,
            output_path=output_path,
            retry_context=retry_context
        )
        
        self.manager.save_and_log_states(prompt, "assembler_prompt.txt")
        return prompt
    
    def _get_iteration_guidance(self, iteration_type: str = None) -> str:
        """Get iteration-specific assembly guidance."""
        if iteration_type == "traditional":
            return """
This iteration focuses on Traditional ML algorithms (XGBoost, LightGBM, CatBoost):
- Ensure proper handling of categorical variables for tree-based models
"""
        elif iteration_type == "custom_nn":
            return """
This iteration focuses on Custom Neural Networks:
- Verify proper neural network architecture implementation
- Ensure training loops with validation monitoring are correctly set up
- Check that loss functions and optimizers are appropriate
- Validate proper batch processing and data loading
- Ensure model checkpointing and early stopping are implemented
"""
        elif iteration_type == "pretrained":
            return """
This iteration focuses on Pretrained Models:
- Verify transfer learning implementation with correct layer freezing
- Check that fine-tuning is properly configured
- Validate model-specific preprocessing is maintained
- Ensure proper adaptation for the target task
"""
        else:
            return ""

    def parse(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response
        
        self.manager.save_and_log_states(code, "final_assembled_code.py")
        return code
