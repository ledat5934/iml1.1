# src/iML/agents/modeling_coder_agent.py
import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts import ModelingCoderPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class ModelingCoderAgent(BaseAgent):
    """
    Agent to create modeling code.
    It only generates code once and does not execute it.
    """
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, **kwargs):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="modeling_coder",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = ModelingCoderPrompt(
            manager=manager, 
            llm_config=self.llm_config
        )

    def __call__(self, iteration_type=None) -> Dict[str, Any]:
        """
        Agent for generating code based on training requirements.
        """
        self.manager.log_agent_start("Starting modeling code generation...")

        guideline = self.manager.guideline
        description = self.manager.description_analysis
        preprocessing_code = self.manager.preprocessing_code
        
        if not preprocessing_code:
            logger.error("Preprocessing code not found. Cannot continue.")
            return {"status": "failed", "error": "Preprocessing code not available."}

        # 1. Generate modeling code
        prompt = self.prompt_handler.build(
            guideline=guideline,
            description=description,
            preprocessing_code=preprocessing_code,
            iteration_type=iteration_type,
        )
        
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(
            content=response,
            save_name="modeling_coder_raw_response.txt",
        )

        modeling_code = self.prompt_handler.parse(response)

        self.manager.log_agent_end("Completed modeling code generation.")
        return {"status": "success", "code": modeling_code}