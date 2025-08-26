# src/iML/agents/assembler_agent.py
import logging
import os
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts import AssemblerPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class AssemblerAgent(BaseAgent):
    """
    Agent to assemble, finalize, execute and fix final code.
    """
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, max_retries: int = 5):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="assembler",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = AssemblerPrompt(
            manager=manager, 
            llm_config=self.llm_config
        )
        self.max_retries = max_retries

    def __call__(self) -> Dict[str, Any]:
        """
        Assemble, execute and retry final code until successful.
        """
        self.manager.log_agent_start("Starting assembly and testing of final code...")

        preprocessing_code = self.manager.preprocessing_code
        modeling_code = self.manager.modeling_code
        description = self.manager.description_analysis

        if not preprocessing_code or not modeling_code:
            error = "Preprocessing or modeling code not available."
            logger.error(error)
            return {"status": "failed", "error": error}

        # Combine initial code
        combined_code = preprocessing_code + "\n\n" + modeling_code
        submission_path = os.path.join(self.manager.output_folder, "submission.csv")
        error_message = None

        for attempt in range(self.max_retries):
            logger.info(f"Assembly and execution attempt {attempt + 1}/{self.max_retries}...")

            # 1. Assemble/Fix code
            # On first attempt, error_message is None, LLM will just assemble.
            # In subsequent attempts, LLM will fix errors.
            prompt = self.prompt_handler.build(
                original_code=combined_code,
                output_path=submission_path,
                description=description,
                error_message=error_message
            )

            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(
                content=response,
                save_name=f"assembler_raw_response_attempt_{attempt + 1}.txt",
            )

            final_code = self.prompt_handler.parse(response)
            
            # 2. Execute final code
            execution_result = self.manager.execute_code(final_code, "assembler", attempt + 1)
            
            if execution_result["success"]:
                logger.info("Final code executed successfully!")
                logger.info(f"Submission file created at: {submission_path}")
                self.manager.save_and_log_states(final_code, "final_executable_code.py")
                self.manager.log_agent_end("Completed assembly and execution of code.")
                return {"status": "success", "code": final_code, "submission_path": submission_path}
            else:
                error_message = execution_result["stderr"]
                logger.warning(f"Code execution failed on attempt {attempt + 1}. Error: {error_message}")
                self.manager.save_and_log_states(
                    f"---ATTEMPT {attempt+1}---\nCODE:\n{final_code}\n\nERROR:\n{error_message}",
                    f"assembler_attempt_{attempt+1}_failed.log"
                )
                # Update combined_code so next iteration LLM will fix the latest error version
                combined_code = final_code

        logger.error(f"Unable to generate working code after {self.max_retries} attempts.")
        self.manager.log_agent_end("Code assembly failed.")
        return {"status": "failed", "error": "Exceeded maximum retry attempts to generate code."}
