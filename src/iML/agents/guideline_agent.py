# src/iML/agents/guideline_agent.py
import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts.guideline_prompt import GuidelinePrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class GuidelineAgent(BaseAgent):
    """
    This agent creates a detailed guideline to solve the problem,
    based on description information and data profiling results.
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)
        
        self.llm_config = llm_config
        self.prompt_template = prompt_template

        self.prompt_handler = GuidelinePrompt(
            llm_config=self.llm_config,
            manager=self.manager,
            template=self.prompt_template,
        )

        # Initialize LLM
        self.llm = init_llm(
            llm_config=self.llm_config,
            agent_name="guideline_agent",
            multi_turn=self.llm_config.get('multi_turn', False)
        )

    def __call__(self) -> Dict[str, Any]:
        """
        Execute agent to create guideline.
        """
        self.manager.log_agent_start("GuidelineAgent: Starting guideline generation...")

        description_analysis = self.manager.description_analysis
        profiling_result = self.manager.profiling_result

        if not description_analysis or "error" in description_analysis:
            logger.error("GuidelineAgent: description_analysis is missing.")
            return {"error": "description_analysis not available."}
        
        if not profiling_result or "error" in profiling_result:
            logger.error("GuidelineAgent: profiling_result is missing.")
            return {"error": "profiling_result not available."}

        # Build prompt
        prompt = self.prompt_handler.build(
            description_analysis=description_analysis,
            profiling_result=profiling_result
        )

        # Call LLM
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(response, "guideline_raw_response.txt")

        # Analyze results
        guideline = self.prompt_handler.parse(response)

        self.manager.log_agent_end("GuidelineAgent: Guideline generation COMPLETED.")
        return guideline
