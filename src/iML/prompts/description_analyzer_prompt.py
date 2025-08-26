# src/iML/prompts/description_analyzer_prompt.py
import json
import os
from .base_prompt import BasePrompt
from typing import Dict, Any 

class DescriptionAnalyzerPrompt(BasePrompt):

    def default_template(self) -> str:
        return """
You are an expert AI assistant specializing in analyzing Kaggle competition descriptions. Your task is to read the provided text and extract key information into a specific JSON structure.
The output MUST be a valid JSON object and nothing else. Do not include any explanatory text before or after the JSON.
Extract the following information:
- "name": Dataset name
- "input_data": A description of the primary input data for the model.
- "output_data": A description of the expected output format from the model.
- "task": A summary of the main objective or task of the competition.
- "data file description": A dictionary where the keys are relative path to the file (e.g., "train.csv", "test/test.csv") and the values are their descriptions.
- "link to the dataset": A list containing the filenames and folders of the core data files (like train, test, sample submission). Do NOT invent or guess full paths.
## EXAMPLE:
### INPUT TEXT:

\"\"\"
Welcome to the 'Paddy Disease Classification' challenge! The goal is to classify diseases in rice paddy images. The input data consists of images of rice plants (JPG files) from the `train_images` folder. Your model should output a class label for one of ten possible diseases. The dataset includes `train.csv` which maps image IDs to their labels, `test_images` for prediction, and `sample_submission.csv` for the required format.
\"\"\"

### OUTPUT JSON:

{{
    "name": "paddy_disease_classification",
    "input_data": "The input data consists of images of rice plants (JPG files).",
    "output_data": "The model should output a class label corresponding to one of ten possible diseases.",
    "task": "The main goal is to build a model that can classify diseases in rice paddy images.",
    "data file description": {{
        "train.csv": "Maps image IDs to their respective disease labels.",
        "train_images": "A folder containing the training images as JPG files.",
        "test_images": "A folder containing the test images for which predictions are required.",
        "sample_submission.csv": "An example file showing the required submission format."
    }},
    "link to the dataset": ["train.csv", "train_images", "test_images", "sample_submission.csv"]
}}
## END OF EXAMPLE. NOW, PROCESS THE FOLLOWING TEXT:
### INPUT TEXT:
\"\"\"
{description}
{directory_structure}
\"\"\"
### OUTPUT JSON:
"""

    # Fixed build method to be correct
    def build(self, description: str, directory_structure: str) -> str:
        """
        Build complete prompt from template and input values.
        """

        prompt = self.template.format(
            description=description,
            directory_structure=directory_structure
        )
        self.manager.save_and_log_states(
            content=prompt, 
            save_name="description_analyzer_prompt.txt"
        )
        return prompt

    # Fixed parse method to be correct
    def parse(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON object.
        """
        try:
            # Clean and parse JSON string from response
            clean_response = response.strip().replace("```json", "").replace("```", "")
            parsed_response = json.loads(clean_response)
        except json.JSONDecodeError as e:
            self.manager.logger.error(f"Failed to parse JSON from LLM response: {e}")
            parsed_response = {"error": "Invalid JSON response from LLM", "raw_response": response}

        # Process additional full path if available
        if "link to the dataset" in parsed_response and isinstance(parsed_response["link to the dataset"], list):
            dataset_path = self.manager.input_data_folder
            file_names = parsed_response["link to the dataset"]
            full_paths = [os.path.join(dataset_path, fname).replace("\\", "/") for fname in file_names]
            parsed_response["link to the dataset"] = full_paths

        self.manager.save_and_log_states(
            content=json.dumps(parsed_response, indent=2), 
            save_name="description_analyzer_response.json"
        )
        return parsed_response