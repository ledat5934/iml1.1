# src/iML/agents/description_analyzer_agent.py

from .base_agent import BaseAgent
from ..prompts.description_analyzer_prompt import DescriptionAnalyzerPrompt
from ..utils.file_io import get_directory_structure
from .utils import init_llm

import logging
import os
from pathlib import Path

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)

class DescriptionAnalyzerAgent(BaseAgent):
    """
    Agent for analyzing the project description and directory structure to understand the task.
    Agent Input: Path to the dataset
    Agent Output: A structured analysis of the ML task.
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)

        self.description_analyzer_llm_config = llm_config
        self.description_analyzer_prompt_template = prompt_template

        # Initialize the corresponding prompt handler
        self.prompt_handler = DescriptionAnalyzerPrompt(
            llm_config=self.description_analyzer_llm_config,
            manager=self.manager,
            template=self.description_analyzer_prompt_template,
        )

        if self.description_analyzer_llm_config.multi_turn:
            self.llm = init_llm(
                llm_config=self.description_analyzer_llm_config,
                agent_name="description_analyzer",
            )

    def __call__(self):
        """
        Executes the description analysis logic.
        """
        self.manager.log_agent_start("DescriptionAnalyzerAgent: Starting Description Analysis...")

        dataset_path = self.manager.input_data_folder
        description_file_path = Path(dataset_path) / "description.txt"

        # Step 1: Read description file
        if not description_file_path.exists():
            logger.error(f"Description file not found at: {description_file_path}")
            return {"error": f"description.txt not found in {dataset_path}"}
        
        with open(description_file_path, "r", encoding="utf-8") as desc_file:
            description = desc_file.read()

        # Step 2: Get directory structure
        directory_structure = get_directory_structure(dataset_path)

        # Step 3: Build the prompt
        prompt = self.prompt_handler.build(
            description=description,
            directory_structure=directory_structure
        )

        # Step 4: Initialize LLM if not already done (for single-turn)
        if not self.description_analyzer_llm_config.multi_turn:
            self.llm = init_llm(
                llm_config=self.description_analyzer_llm_config,
                agent_name="description_analyzer",
                multi_turn=self.description_analyzer_llm_config.multi_turn,
            )

        # Step 5: Call the LLM
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(
            content=response, 
            save_name="description_analyzer_raw_response.txt"
        )

        # Step 6: Parse the response
        analysis_result = self.prompt_handler.parse(response)

        self.manager.log_agent_end("DescriptionAnalyzerAgent: Description Analysis COMPLETED.")
        return analysis_result