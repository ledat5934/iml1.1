# src/iML/agents/comparison_agent.py
import logging
import json
from typing import Dict, Any, List
from pathlib import Path

from .base_agent import BaseAgent
from ..prompts.comparison_prompt import ComparisonPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class ComparisonAgent(BaseAgent):
    """
    Agent to intelligently compare and rank multiple iteration results using LLM.
    """
    
    def __init__(self, config: Dict, manager: Any, llm_config: Dict):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="comparison",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = ComparisonPrompt(
            manager=manager, 
            llm_config=self.llm_config
        )

    def __call__(self, iteration_results: List[Dict[str, Any]], 
                 original_task_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare iterations and select the best one using LLM intelligence.
        
        Args:
            iteration_results: List of iteration results with extracted data
            original_task_description: Original task description for context
            
        Returns:
            Dict containing comparison analysis and best iteration selection
        """
        self.manager.log_agent_start("ComparisonAgent: Starting intelligent iteration comparison...")
        
        if not iteration_results:
            logger.error("No iteration results provided for comparison")
            return {"error": "No iteration results to compare"}
        
        # Build prompt with all iteration data
        prompt = self.prompt_handler.build(
            iteration_results=iteration_results,
            task_description=original_task_description
        )
        
        # Get LLM analysis
        response = self.llm.assistant_chat(prompt)
        self.manager.save_and_log_states(
            content=response,
            save_name="comparison_raw_response.txt",
        )
        
        # Parse LLM response
        comparison_result = self.prompt_handler.parse(response)
        
        if "error" not in comparison_result:
            logger.info(f"LLM selected best iteration: {comparison_result.get('best_iteration', {}).get('name', 'Unknown')}")
            logger.info(f"Selection reasoning: {comparison_result.get('reasoning_summary', 'No reasoning provided')}")
        else:
            logger.error(f"Comparison failed: {comparison_result['error']}")
        
        self.manager.log_agent_end("ComparisonAgent: Intelligent comparison completed.")
        return comparison_result


class IterationResultExtractor:
    """Extract comprehensive results from iteration folders for LLM analysis."""
    
    def __init__(self):
        # No more regex patterns needed - LLM will extract scores from raw output
        pass
    
    def extract_execution_stats(self, iteration_path: Path) -> Dict[str, Any]:
        """Extract execution statistics and metadata."""
        stats = {
            "execution_time": None,
            "memory_usage": None,
            "num_attempts": 0,
            "final_attempt": None,
            "error_count": 0,
            "warnings": []
        }
        
        try:
            attempts_dir = iteration_path / "attempts" / "assembler"
            if attempts_dir.exists():
                attempt_dirs = [d for d in attempts_dir.iterdir() if d.is_dir() and d.name.startswith("attempt_")]
                stats["num_attempts"] = len(attempt_dirs)
                
                if attempt_dirs:
                    # Get the highest numbered attempt
                    final_attempt_num = max([int(d.name.split("_")[1]) for d in attempt_dirs])
                    stats["final_attempt"] = final_attempt_num
                    
                    # Check for errors in attempts
                    for attempt_dir in attempt_dirs:
                        stderr_file = attempt_dir / "stderr.txt"
                        if stderr_file.exists():
                            try:
                                with open(stderr_file, 'r', encoding='utf-8') as f:
                                    stderr_content = f.read()
                                    if stderr_content.strip():
                                        stats["error_count"] += 1
                            except:
                                pass
        except Exception as e:
            logger.warning(f"Could not extract execution stats from {iteration_path}: {e}")
        
        return stats
    
    def extract_code_complexity(self, iteration_path: Path) -> Dict[str, Any]:
        """Extract code complexity metrics."""
        complexity = {
            "lines_of_code": 0,
            "imports_count": 0,
            "functions_count": 0,
            "model_type": "unknown"
        }
        
        try:
            final_code_path = iteration_path / "states" / "final_executable_code.py"
            if final_code_path.exists():
                with open(final_code_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                lines = code_content.split('\n')
                complexity["lines_of_code"] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                complexity["imports_count"] = len([l for l in lines if l.strip().startswith('import') or l.strip().startswith('from')])
                complexity["functions_count"] = len([l for l in lines if l.strip().startswith('def ')])
                
                # Detect model type
                code_lower = code_content.lower()
                if 'xgboost' in code_lower or 'lightgbm' in code_lower or 'catboost' in code_lower:
                    complexity["model_type"] = "traditional_ml"
                elif 'torch' in code_lower or 'nn.module' in code_lower:
                    complexity["model_type"] = "custom_neural_network"
                elif 'transformers' in code_lower or 'pretrained' in code_lower:
                    complexity["model_type"] = "pretrained_model"
                    
        except Exception as e:
            logger.warning(f"Could not extract code complexity from {iteration_path}: {e}")
        
        return complexity
    
    def extract_from_iteration_folder(self, iteration_path: str) -> Dict[str, Any]:
        """Extract comprehensive results from an iteration folder."""
        iteration_path = Path(iteration_path)
        
        result = {
            "iteration_name": iteration_path.name,
            "status": "success",
            "error": None,
            "scores": {},
            "execution_stats": {},
            "code_complexity": {},
            "output_files": [],
            "stdout_excerpt": "",
            "stderr_excerpt": ""
        }
        
        try:
            # Check basic success criteria
            submission_file = iteration_path / "submission.csv"
            if not submission_file.exists():
                result["status"] = "failed"
                result["error"] = "No submission.csv found"
                return result
            
            result["output_files"].append("submission.csv")
            
            # Extract scores from stdout
            stdout_files = [
                iteration_path / "attempts" / "assembler" / f"attempt_{i}" / "stdout.txt"
                for i in range(1, 6)  # Check up to 5 attempts
            ]
            
            all_stdout = ""
            for stdout_file in stdout_files:
                if stdout_file.exists():
                    try:
                        with open(stdout_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            all_stdout += content + "\n"
                    except Exception as e:
                        logger.warning(f"Could not read {stdout_file}: {e}")
            
            if all_stdout:
                # Keep full stdout for LLM to analyze - no need for regex parsing
                result["full_stdout"] = all_stdout
                # Keep last 1000 chars for summary display
                result["stdout_excerpt"] = all_stdout[-1000:] if len(all_stdout) > 1000 else all_stdout
            
            # Extract stderr excerpt  
            stderr_files = [f.replace("stdout.txt", "stderr.txt") for f in stdout_files]
            all_stderr = ""
            for stderr_file in stderr_files:
                if Path(stderr_file).exists():
                    try:
                        with open(stderr_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                all_stderr += content + "\n"
                    except:
                        pass
            
            if all_stderr:
                # Keep last 200 chars of stderr for context
                result["stderr_excerpt"] = all_stderr[-200:] if len(all_stderr) > 200 else all_stderr
            
            # Extract execution stats
            result["execution_stats"] = self.extract_execution_stats(iteration_path)
            
            # Extract code complexity
            result["code_complexity"] = self.extract_code_complexity(iteration_path)
            
            # Check for other output files
            for file_path in iteration_path.rglob("*.csv"):
                if file_path != submission_file:
                    result["output_files"].append(str(file_path.relative_to(iteration_path)))
                    
            # Remove the scores requirement check - LLM will extract scores from stdout
                
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Error extracting results from {iteration_path}: {e}")
        
        return result
