import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts import PreprocessingCoderPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class PreprocessingCoderAgent(BaseAgent):
    """
    Agent to create and execute preprocessing code.
    It has a retry loop to generate and validate code until it runs successfully or runs out of retries.
    """
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, max_retries: int = 5):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="preprocessing_coder",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = PreprocessingCoderPrompt(
            manager=manager, 
            llm_config=self.llm_config
        )
        self.max_retries = max_retries

    def __call__(self) -> Dict[str, Any]:
        """
        Generate, execute and retry preprocessing code until successful or maximum retries exceeded.
        """
        self.manager.log_agent_start("Starting preprocessing code generation...")

        guideline = self.manager.guideline
        description = self.manager.description_analysis
        
        code_to_execute = None
        error_message = None
        
        for attempt in range(self.max_retries):
            logger.info(f"Code generation attempt {attempt + 1}/{self.max_retries}...")

            # 1. Generate code
            prompt = self.prompt_handler.build(
                guideline=guideline,
                description=description,
                previous_code=code_to_execute,
                error_message=error_message
            )

            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(
                content=response,
                save_name=f"preprocessing_coder_raw_response_attempt_{attempt + 1}.txt",
            )
            
            code_to_execute = self.prompt_handler.parse(response)

            # 2. Execute code
            execution_result = self.manager.execute_code(code_to_execute, "preprocessing", attempt + 1)
            
            # 3. Check results
            if execution_result["success"]:
                logger.info("Preprocessing code executed successfully!")
                self.manager.save_and_log_states(code_to_execute, "final_preprocessing_code.py")
                self.manager.log_agent_end("Completed preprocessing code generation.")
                return {"status": "success", "code": code_to_execute}
            else:
                error_message = execution_result["stderr"]
                logger.warning(f"Code execution failed on attempt {attempt + 1}. Error: {error_message}")
                self.manager.save_and_log_states(
                    f"---ATTEMPT {attempt+1}---\nCODE:\n{code_to_execute}\n\nERROR:\n{error_message}",
                    f"preprocessing_attempt_{attempt+1}_failed.log"
                )

        logger.error(f"Unable to generate working preprocessing code after {self.max_retries} attempts.")
        self.manager.log_agent_end("Preprocessing code generation failed.")
        return {"status": "failed", "error": "Exceeded maximum retry attempts to generate code."}
