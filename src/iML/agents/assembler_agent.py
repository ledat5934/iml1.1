# src/iML/agents/assembler_agent.py
import logging
import os
import re
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from ..prompts import AssemblerPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class AssemblerAgent(BaseAgent):
    """
    Agent to assemble, finalize, execute and fix final code.
    """
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, max_retries: int = 10):
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

    def _extract_performance_metrics(self, stdout_content: str) -> Dict[str, float]:
        """Extract performance metrics from assembler execution stdout."""
        metrics = {}
        
        # Common metric patterns
        patterns = {
            'validation_score': r'Validation Score:\s*([0-9]*\.?[0-9]+)',
            'cv_score': r'CV Score:\s*([0-9]*\.?[0-9]+)',
            'mean_cv_score': r'Mean CV Score:\s*([0-9]*\.?[0-9]+)',
            'accuracy': r'Accuracy:\s*([0-9]*\.?[0-9]+)',
            'f1_score': r'F1[- ]?Score:\s*([0-9]*\.?[0-9]+)',
            'rmse': r'RMSE:\s*([0-9]*\.?[0-9]+)',
            'mae': r'MAE:\s*([0-9]*\.?[0-9]+)',
            'r2_score': r'R2[- ]?Score:\s*([0-9]*\.?[0-9]+)',
            'auc': r'AUC:\s*([0-9]*\.?[0-9]+)',
            'precision': r'Precision:\s*([0-9]*\.?[0-9]+)',
            'recall': r'Recall:\s*([0-9]*\.?[0-9]+)',
        }
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, stdout_content, re.IGNORECASE)
            if matches:
                try:
                    # Take the last occurrence (most recent score)
                    metrics[metric_name] = float(matches[-1])
                except ValueError:
                    continue
        
        return metrics

    def _get_primary_score(self, metrics: Dict[str, float]) -> Optional[float]:
        """Get the primary performance score from metrics."""
        # Priority order for score selection
        score_priority = ['mean_cv_score', 'cv_score', 'validation_score', 'accuracy', 
                         'f1_score', 'auc', 'r2_score', 'precision', 'recall']
        
        for score_name in score_priority:
            if score_name in metrics:
                return metrics[score_name]
        
        return None

    def __call__(self, iteration_type=None) -> Dict[str, Any]:
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
                error_message=error_message,
                iteration_type=iteration_type
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
                
                # Extract and log performance metrics
                stdout_content = execution_result.get("stdout", "")
                if stdout_content:
                    metrics = self._extract_performance_metrics(stdout_content)
                    primary_score = self._get_primary_score(metrics)
                    
                    # Get iteration type for clear logging
                    iteration_name = os.path.basename(self.manager.output_folder)
                    
                    logger.info("🔥" + "="*60)
                    logger.info(f"📊 ASSEMBLER EXECUTION COMPLETED: {iteration_name}")
                    logger.info("🔥" + "="*60)
                    
                    if metrics:
                        logger.info(f"📈 Extracted performance metrics: {metrics}")
                        if primary_score is not None:
                            logger.info(f"🏆 Primary performance score: {primary_score:.4f}")
                        else:
                            logger.warning("⚠️ No primary score identified from metrics")
                    else:
                        logger.warning("⚠️ No performance metrics extracted from execution output")
                    
                    logger.info(f"✅ Original submission created: submission.csv")
                    logger.info(f"📄 Original submission file: {iteration_name}/submission.csv")
                    logger.info("🔥" + "="*60)
                
                self.manager.log_agent_end("Completed assembly and execution of code.")
                return {"status": "success", "code": final_code, "submission_path": submission_path, 
                       "metrics": metrics if 'metrics' in locals() else {}}
            else:
                error_message = execution_result["stderr"]
                last_10_lines = error_message.split('\n')[-10:]
                error_to_log = '\n'.join(last_10_lines)
                logger.warning(f"Code execution failed on attempt {attempt + 1}. Error: {error_message}")
                self.manager.save_and_log_states(
                    f"---ATTEMPT {attempt+1}---\nCODE:\n{final_code}\n\nERROR:\n{error_to_log}",
                    f"assembler_attempt_{attempt+1}_failed.log"
                )
                # Update combined_code so next iteration LLM will fix the latest error version
                combined_code = final_code

        logger.error(f"Unable to generate working code after {self.max_retries} attempts.")
        self.manager.log_agent_end("Code assembly failed.")
        return {"status": "failed", "error": "Exceeded maximum retry attempts to generate code."}
