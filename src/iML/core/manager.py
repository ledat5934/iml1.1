import logging
import os
import uuid
import subprocess
import json
import shutil
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from ..agents import (
    DescriptionAnalyzerAgent,
    ProfilingAgent,
    ProfilingSummarizerAgent,
    ModelRetrieverAgent,
    GuidelineAgent,
    PreprocessingCoderAgent,
    ModelingCoderAgent,
    AssemblerAgent,
    ComparisonAgent,
)
from ..agents.comparison_agent import IterationResultExtractor
from ..llm import ChatLLMFactory
from ..agents.hyperparameter_tuning_agent import HyperparameterTuningAgent

# Basic configuration
logging.basicConfig(level=logging.INFO)

# Create a logger
logger = logging.getLogger(__name__)


class Manager:
    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: str,
    ):
        """Initialize Manager with required paths and config from YAML file.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config_path: Path to YAML configuration file
        """
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder
        self.config = config

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.description_analyzer_agent = DescriptionAnalyzerAgent(
            config=config,
            manager=self,
            llm_config=self.config.description_analyzer,
        )
        self.profiling_agent = ProfilingAgent(
            config=config,
            manager=self,
        )
        self.profiling_summarizer_agent = ProfilingSummarizerAgent(
            config=config,
            manager=self,
            llm_config=self.config.profiling_summarizer,
        )
        self.model_retriever_agent = ModelRetrieverAgent(
            config=config,
            manager=self,
        )
        self.guideline_agent = GuidelineAgent(
            config=config,
            manager=self,
            llm_config=self.config.guideline_generator,
        )
        self.preprocessing_coder_agent = PreprocessingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.preprocessing_coder,
        )
        self.modeling_coder_agent = ModelingCoderAgent(
            config=config,
            manager=self,
            llm_config=self.config.modeling_coder,
        )
        self.assembler_agent = AssemblerAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,
        )
        # Initialize hyperparameter tuning agent
        self.hyperparameter_tuning_agent = HyperparameterTuningAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler
        )
        self.comparison_agent = ComparisonAgent(
            config=config,
            manager=self,
            llm_config=self.config.assembler,  # Using same LLM config as assembler
        )

        self.context = {
            "input_data_folder": input_data_folder,
            "output_folder": output_folder,
            
        }

    def run_pipeline_partial(self, stop_after="guideline"):
        """Run pipeline up to a specific checkpoint."""
        logger.info(f"Starting partial AutoML pipeline (stop after: {stop_after})...")

        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return False
        logger.info(f"Analysis result: {analysis_result}")
        self.description_analysis = analysis_result

        if stop_after == "description":
            logger.info("Pipeline stopped after description analysis.")
            return True

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return False
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        if stop_after == "profiling":
            logger.info("Pipeline stopped after profiling.")
            return True

        # Step 3a: Summarize profiling via LLM
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return False
        self.profiling_summary = profiling_summary

        # Step 3b: Retrieve pretrained model suggestions
        model_suggestions = self.model_retriever_agent()
        self.model_suggestions = model_suggestions

        if stop_after == "pre-guideline":
            # Save the default prompt template for editing
            self.save_default_guideline_prompt_template()
            logger.info("Pipeline stopped before guideline generation.")
            logger.info("You can now edit the guideline prompt template and resume from guideline generation.")
            return True

        # Step 3c: Run guideline agent
        guideline = self.guideline_agent()
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return False
        self.guideline = guideline
        logger.info("Guideline generated successfully.")

        if stop_after == "guideline":
            logger.info("Pipeline stopped after guideline generation.")
            logger.info("You can now manually edit the guideline in the states folder.")
            return True

        logger.info("Partial AutoML pipeline completed successfully!")
        return True

    def load_checkpoint_state(self):
        """Load previously saved checkpoint state."""
        import json
        import os
        
        states_dir = os.path.join(self.output_folder, "states")
        
        # Debug: List all files in states directory
        if os.path.exists(states_dir):
            files_in_states = os.listdir(states_dir)
            logger.info(f"Files found in states directory: {files_in_states}")
        else:
            logger.warning(f"States directory does not exist: {states_dir}")
            return
        
        # Load description analysis
        desc_file = os.path.join(states_dir, "description_analyzer_response.json")
        if os.path.exists(desc_file):
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    self.description_analysis = json.load(f)
                logger.info("Loaded description analysis from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load description analysis: {e}")
        else:
            logger.warning(f"Description analysis file not found: {desc_file}")
            # Initialize as None so we can check later
            self.description_analysis = None
        
        # Load profiling result
        prof_file = os.path.join(states_dir, "profiling_result.json")
        if os.path.exists(prof_file):
            with open(prof_file, 'r', encoding='utf-8') as f:
                self.profiling_result = json.load(f)
            logger.info("Loaded profiling result from checkpoint")
        
        # Load profiling summary  
        prof_sum_file = os.path.join(states_dir, "profiling_summary.json")
        if os.path.exists(prof_sum_file):
            with open(prof_sum_file, 'r', encoding='utf-8') as f:
                self.profiling_summary = json.load(f)
            logger.info("Loaded profiling summary from checkpoint")
        
        # Load model suggestions
        model_file = os.path.join(states_dir, "model_retrieval.json")
        if os.path.exists(model_file):
            with open(model_file, 'r', encoding='utf-8') as f:
                self.model_suggestions = json.load(f)
            logger.info("Loaded model suggestions from checkpoint")
        
        # Load guideline (might be manually edited)
        guideline_file = os.path.join(states_dir, "guideline_response.json")
        if os.path.exists(guideline_file):
            with open(guideline_file, 'r', encoding='utf-8') as f:
                self.guideline = json.load(f)
            logger.info("Loaded guideline from checkpoint")

    def resume_pipeline_from_checkpoint(self, start_from="preprocessing"):
        """Resume pipeline from a specific checkpoint."""
        logger.info(f"Resuming AutoML pipeline from: {start_from}...")
        
        # Load previous state
        self.load_checkpoint_state()
        
        # Validate required states are loaded
        if not hasattr(self, 'description_analysis') or self.description_analysis is None:
            logger.error("Cannot resume: description_analysis not found or is None")
            logger.error("Make sure you have run the pipeline at least until the description analysis step")
            return False
        
        # For resume from guideline, we don't need existing guideline
        if start_from != "guideline" and (not hasattr(self, 'guideline') or self.guideline is None):
            logger.error(f"Cannot resume from {start_from}: guideline not found")
            logger.error("For this resume point, you need to have run until guideline generation")
            return False

        if start_from == "guideline":
            # Load custom prompt template if available
            self.update_guideline_prompt_template()
            
            # Check if we have necessary data for guideline generation
            if not hasattr(self, 'profiling_result') or not hasattr(self, 'model_suggestions'):
                logger.warning("Missing profiling or model suggestions data. Running those steps first...")
                
                # Re-run profiling if needed
                if not hasattr(self, 'profiling_result'):
                    profiling_result = self.profiling_agent()
                    if "error" in profiling_result:
                        logger.error(f"Data profiling failed: {profiling_result['error']}")
                        return False
                    self.profiling_result = profiling_result
                
                # Re-run profiling summary if needed
                if not hasattr(self, 'profiling_summary'):
                    profiling_summary = self.profiling_summarizer_agent()
                    if "error" in profiling_summary:
                        logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
                        return False
                    self.profiling_summary = profiling_summary
                
                # Re-run model retrieval if needed
                if not hasattr(self, 'model_suggestions'):
                    model_suggestions = self.model_retriever_agent()
                    self.model_suggestions = model_suggestions
            
            # Re-run guideline generation (useful after editing prompt)
            guideline = self.guideline_agent()
            if "error" in guideline:
                logger.error(f"Guideline generation failed: {guideline['error']}")
                return False
            self.guideline = guideline
            logger.info("Guideline regenerated successfully.")

        if start_from in ["guideline", "preprocessing"]:
            # Step 4: Run Preprocessing Coder Agent
            preprocessing_code_result = self.preprocessing_coder_agent()
            if preprocessing_code_result.get("status") == "failed":
                logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
                return False
            self.preprocessing_code = preprocessing_code_result.get("code")
            logger.info("Preprocessing code generated and validated successfully.")

        if start_from in ["guideline", "preprocessing", "modeling"]:
            # Step 5: Run Modeling Coder Agent
            modeling_code_result = self.modeling_coder_agent()
            if modeling_code_result.get("status") == "failed":
                logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
                return False
            self.modeling_code = modeling_code_result.get("code")
            logger.info("Modeling code generated successfully.")

        if start_from in ["guideline", "preprocessing", "modeling", "assembler"]:
            # Step 6: Run Assembler Agent
            assembler_result = self.assembler_agent()
            if assembler_result.get("status") == "failed":
                logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
                return False
            self.assembled_code = assembler_result.get("code")
            logger.info("Initial script generated and executed successfully.")

        logger.info("AutoML pipeline completed successfully!")
        return True

    def update_guideline_prompt_template(self, new_template: str = None):
        """
        Update the guideline prompt template.
        If new_template is None, it will load from a file if it exists.
        """
        import os
        
        # Try to load from file first
        custom_prompt_file = os.path.join(self.output_folder, "custom_guideline_prompt.txt")
        if new_template is None and os.path.exists(custom_prompt_file):
            with open(custom_prompt_file, 'r', encoding='utf-8') as f:
                new_template = f.read()
            logger.info("Loaded custom guideline prompt from file")
        elif new_template is None:
            logger.info("No custom prompt template provided, using default")
            return
        
        # Update the template
        self.guideline_agent.prompt_handler.template = new_template
        logger.info("Guideline prompt template updated")
        
        # Save the template for reference
        self.save_and_log_states(new_template, "guideline_prompt_template_used.txt")

    def save_default_guideline_prompt_template(self):
        """Save the default guideline prompt template for editing."""
        default_template = self.guideline_agent.prompt_handler.default_template()
        template_file = os.path.join(self.output_folder, "custom_guideline_prompt.txt")
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(default_template)
        
        logger.info(f"Default guideline prompt template saved to: {template_file}")
        logger.info("You can edit this file and resume from guideline generation.")
        return template_file

    def run_pipeline_multi_iteration(self):
        """Run the pipeline with 3 iterations for different algorithm approaches."""
        logger.info("Starting Multi-Iteration AutoML Pipeline...")
        
        # Store original output folder
        original_output_folder = self.output_folder
        
        # Define iterations
        iterations = [
            {
                "name": "traditional",
                "folder": "iteration_1_traditional",
                "description": "Traditional ML algorithms (XGBoost, LightGBM, CatBoost)"
            },
            {
                "name": "custom_nn", 
                "folder": "iteration_2_custom_nn",
                "description": "Custom Neural Networks"
            },
            {
                "name": "pretrained",
                "folder": "iteration_3_pretrained", 
                "description": "Pretrained Models"
            }
        ]
        
        # Run shared analysis steps once
        logger.info("Running shared analysis steps...")
        success = self._run_shared_analysis()
        if not success:
            return
            
        # Run each iteration
        iteration_paths = []
        for i, iteration in enumerate(iterations, 1):
            logger.info(f"=== Starting Iteration {i}: {iteration['description']} ===")
            
            # Create iteration-specific output folder
            iteration_output = os.path.join(original_output_folder, iteration['folder'])
            os.makedirs(iteration_output, exist_ok=True)
            iteration_paths.append(iteration_output)
            
            # Temporarily change output folder for this iteration
            self.output_folder = iteration_output
            
            # Run iteration-specific pipeline
            success = self._run_iteration_pipeline(iteration['name'])
            if not success:
                logger.error(f"Iteration {i} ({iteration['name']}) failed!")
            else:
                logger.info(f"=== Iteration {i} completed successfully ===")
        
        # Restore original output folder
        self.output_folder = original_output_folder
        
        # Extract comprehensive results from all iterations
        logger.info("=== Extracting results from all iterations ===")
        extractor = IterationResultExtractor()
        iteration_results = []
        
        for iteration_path in iteration_paths:
            result = extractor.extract_from_iteration_folder(iteration_path)
            iteration_results.append(result)
            logger.info(f"Extracted results from {result['iteration_name']}: {result['status']}")
        
        # Use LLM to intelligently compare and rank iterations
        logger.info("=== LLM-based Intelligent Iteration Comparison ===")
        comparison_result = self.comparison_agent(
            iteration_results=iteration_results,
            original_task_description=self.description_analysis
        )
        
        if "error" in comparison_result:
            logger.error(f"LLM comparison failed: {comparison_result['error']}")
            logger.info("Falling back to basic selection...")
            best_iteration_name = self._fallback_selection(iteration_results)
        else:
            best_iteration_name = comparison_result.get('best_iteration', {}).get('name')
            
            # Save detailed comparison report
            comparison_file = os.path.join(original_output_folder, "llm_comparison_results.json")
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            logger.info(f"LLM comparison report saved to: {comparison_file}")
        
        # Copy best submission to final_submission folder
        if best_iteration_name:
            # Normalize iteration name to handle case mismatch
            normalized_iteration_name = self._normalize_iteration_name(best_iteration_name, original_output_folder)
            
            best_iteration_path = os.path.join(original_output_folder, normalized_iteration_name)
            logger.info(f"🔍 Copying best submission from: {normalized_iteration_name}")
            if normalized_iteration_name != best_iteration_name:
                logger.info(f"📝 Note: LLM selected '{best_iteration_name}', normalized to '{normalized_iteration_name}'")
            
            success = self._copy_best_submission(best_iteration_path, original_output_folder)
            
            if success:
                logger.info(f"🎉 Best submission successfully copied from {normalized_iteration_name}")
                if "error" not in comparison_result:
                    logger.info(f"🧠 LLM Reasoning: {comparison_result.get('reasoning_summary', 'No reasoning provided')}")
                logger.info(f"📁 Final submission available in: {os.path.join(original_output_folder, 'final_submission')}")
            else:
                logger.error(f"❌ Failed to copy best submission from {normalized_iteration_name}")
                logger.error("🔧 This may be due to missing submission files in the iteration directory")
                logger.error(f"🔧 Checked path: {best_iteration_path}")
        else:
            logger.error("❌ No best iteration selected - this should not happen")
        
        # Print final summary
        successful_count = len([r for r in iteration_results if r.get('status') == 'success'])
        logger.info("🎊" + "="*70)
        logger.info("🎊 MULTI-ITERATION AUTOML PIPELINE COMPLETED")
        logger.info("🎊" + "="*70)
        logger.info(f"📈 Summary: {successful_count}/{len(iterations)} iterations successful")
        
        if best_iteration_name:
            # Extract submission type from metadata
            final_submission_path = os.path.join(original_output_folder, "final_submission")
            metadata_file = os.path.join(final_submission_path, "selection_metadata.json")
            
            submission_identifier = "unknown"
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        submission_identifier = metadata.get("submission_identifier", "unknown")
                except:
                    pass
            
            logger.info(f"🏆 FINAL BEST SUBMISSION: {submission_identifier}")
            logger.info(f"📂 Source iteration: {best_iteration_name}")
            logger.info(f"📁 Final submission location: final_submission/submission.csv")
        
        logger.info("🎊" + "="*70)
        
        logger.info("Multi-Iteration AutoML Pipeline completed!")
    
    def _fallback_selection(self, iteration_results: List[Dict]) -> str:
        """Fallback selection method when LLM comparison fails."""
        # Simple fallback: prefer successful iterations in priority order
        priority_order = ["iteration_3_pretrained", "iteration_2_custom_nn", "iteration_1_traditional"]
        
        successful_iterations = [
            r for r in iteration_results 
            if r.get('status') == 'success'
        ]
        
        if not successful_iterations:
            logger.warning("No successful iterations found for fallback selection")
            return None
        
        logger.info(f"🔄 Fallback selection from {len(successful_iterations)} successful iterations")
        
        # Select based on priority order (case-insensitive)
        for preferred_name in priority_order:
            for iteration in successful_iterations:
                iteration_name = iteration.get('iteration_name', '').lower()
                if preferred_name.lower() in iteration_name:
                    actual_name = iteration['iteration_name']
                    logger.info(f"🎯 Fallback selected: {actual_name} (matched {preferred_name})")
                    return actual_name
        
        # If no match, select first successful
        first_successful = successful_iterations[0]['iteration_name']
        logger.info(f"⚠️ Fallback selected first successful: {first_successful}")
        return first_successful
    
    def _normalize_iteration_name(self, iteration_name: str, base_output_folder: str) -> Optional[str]:
        """Normalize iteration name to match actual directory name (case-insensitive)."""
        if not iteration_name:
            return None
        
        base_path = Path(base_output_folder)
        if not base_path.exists():
            return iteration_name
        
        # Get all iteration directories
        iteration_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('iteration_')]
        
        # Try exact match first
        target_path = base_path / iteration_name
        if target_path.exists():
            return iteration_name
        
        # Try case-insensitive match
        iteration_name_lower = iteration_name.lower()
        for dir_path in iteration_dirs:
            if dir_path.name.lower() == iteration_name_lower:
                logger.info(f"🔧 Normalized iteration name: {iteration_name} → {dir_path.name}")
                return dir_path.name
        
        # No match found
        logger.warning(f"⚠️ No matching directory found for iteration: {iteration_name}")
        logger.warning(f"Available directories: {[d.name for d in iteration_dirs]}")
        return iteration_name
    
    def _extract_performance_metrics(self, stdout_content: str) -> Dict[str, float]:
        """Extract performance metrics from stdout content."""
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
    
    def _get_submission_performance(self, iteration_path: Path, submission_type: str) -> Tuple[Optional[float], Dict[str, float]]:
        """Get performance score for a specific submission type."""
        
        if submission_type == "original":
            # Look in assembler attempts for original submission performance
            attempts_dir = iteration_path / "attempts" / "assembler"
        elif submission_type == "tuned":
            # Look in hyperparameter_tuning attempts for tuned submission performance
            attempts_dir = iteration_path / "attempts" / "hyperparameter_tuning"
        else:
            return None, {}
        
        if not attempts_dir.exists():
            return None, {}
        
        # Get all stdout content from attempts
        all_stdout = ""
        attempt_dirs = [d for d in attempts_dir.iterdir() if d.is_dir() and d.name.startswith("attempt_")]
        
        for attempt_dir in attempt_dirs:
            stdout_file = attempt_dir / "stdout.txt"
            if stdout_file.exists():
                try:
                    with open(stdout_file, 'r', encoding='utf-8') as f:
                        all_stdout += f.read() + "\n"
                except Exception as e:
                    logger.warning(f"Could not read {stdout_file}: {e}")
        
        # Extract metrics
        metrics = self._extract_performance_metrics(all_stdout)
        
        # Determine the primary score to use for comparison
        primary_score = None
        
        # Priority order for primary score selection
        score_priority = ['mean_cv_score', 'cv_score', 'validation_score', 'accuracy', 'f1_score', 'auc', 'r2_score']
        
        for score_name in score_priority:
            if score_name in metrics:
                primary_score = metrics[score_name]
                break
        
        return primary_score, metrics
    
    def _select_best_submission_file(self, iteration_path: Path) -> Tuple[Optional[Path], str, Dict]:
        """Select the best submission file based on performance metrics."""
        
        original_submission = iteration_path / "submission.csv"
        tuned_submission = iteration_path / "submission_tuned.csv"
        
        # Check which files exist
        original_exists = original_submission.exists()
        tuned_exists = tuned_submission.exists()
        
        if not original_exists and not tuned_exists:
            return None, "none", {"error": "No submission files found"}
        
        if tuned_exists and not original_exists:
            return tuned_submission, "tuned", {"reason": "Only tuned submission available"}
        
        if original_exists and not tuned_exists:
            return original_submission, "original", {"reason": "Only original submission available"}
        
        # Both files exist - compare performance
        original_score, original_metrics = self._get_submission_performance(iteration_path, "original")
        tuned_score, tuned_metrics = self._get_submission_performance(iteration_path, "tuned")
        
        logger.info(f"📊 Performance comparison for {iteration_path.name}:")
        logger.info(f"   Original submission score: {original_score} (metrics: {original_metrics})")
        logger.info(f"   Tuned submission score: {tuned_score} (metrics: {tuned_metrics})")
        
        # Decision logic
        decision_info = {
            "original_score": original_score,
            "tuned_score": tuned_score,
            "original_metrics": original_metrics,
            "tuned_metrics": tuned_metrics
        }
        
        # If we can't get scores for either, fall back to tuned (assuming it's better)
        if original_score is None and tuned_score is None:
            logger.warning("🔍 No performance metrics found for either submission, defaulting to tuned version")
            return tuned_submission, "tuned", {**decision_info, "reason": "No metrics found, defaulting to tuned"}
        
        # If only one has a score, pick that one
        if original_score is None and tuned_score is not None:
            logger.info("🎯 Selecting tuned submission (only one with metrics)")
            return tuned_submission, "tuned", {**decision_info, "reason": "Only tuned has metrics"}
        
        if tuned_score is None and original_score is not None:
            logger.info("📊 Selecting original submission (only one with metrics)")
            return original_submission, "original", {**decision_info, "reason": "Only original has metrics"}
        
        # Both have scores - compare them
        if tuned_score > original_score:
            improvement = tuned_score - original_score
            logger.info(f"🎯 Selecting tuned submission (score: {tuned_score:.4f} vs {original_score:.4f}, +{improvement:.4f})")
            return tuned_submission, "tuned", {**decision_info, "reason": f"Tuned performs better by {improvement:.4f}"}
        else:
            decline = original_score - tuned_score
            logger.info(f"📊 Selecting original submission (score: {original_score:.4f} vs {tuned_score:.4f}, tuned declined by {decline:.4f})")
            return original_submission, "original", {**decision_info, "reason": f"Original performs better by {decline:.4f}"}
    
    def _copy_best_submission(self, source_iteration_path: str, target_folder: str) -> bool:
        """Copy the best submission to final_submission folder based on performance comparison."""
        try:
            source_path = Path(source_iteration_path)
            target_path = Path(target_folder) / "final_submission"
            
            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Select the best submission file based on performance
            best_submission_file, submission_type, decision_info = self._select_best_submission_file(source_path)
            
            if best_submission_file is None:
                error_msg = decision_info.get("error", "Unknown error in submission selection")
                logger.error(f"❌ {error_msg} in {source_path}")
                return False
            
            files_copied = []
            
            # Log the selection decision with detailed format
            reason = decision_info.get("reason", "No reason provided")
            
            # Create detailed submission identifier
            iteration_base = source_path.name
            if "traditional" in iteration_base.lower():
                iter_type = "traditional"
            elif "custom_nn" in iteration_base.lower():
                iter_type = "custom_nn"
            elif "pretrained" in iteration_base.lower():
                iter_type = "pretrained"
            else:
                iter_type = "unknown"
            
            tuning_status = "tuned" if submission_type == "tuned" else "non_tuned"
            best_submission_identifier = f"iter_{iter_type}_{tuning_status}"
            
            logger.info("🏆" + "="*60)
            logger.info(f"🎯 BEST SUBMISSION SELECTED: {best_submission_identifier}")
            logger.info("🏆" + "="*60)
            logger.info(f"📄 File: {best_submission_file.name} ({submission_type})")
            logger.info(f"📋 Selection reason: {reason}")
            
            # Log performance comparison if available
            original_score = decision_info.get("original_score")
            tuned_score = decision_info.get("tuned_score")
            if original_score is not None and tuned_score is not None:
                logger.info(f"📊 Performance comparison:")
                logger.info(f"   Original score: {original_score:.4f}")
                logger.info(f"   Tuned score: {tuned_score:.4f}")
                if tuned_score > original_score:
                    improvement = tuned_score - original_score
                    logger.info(f"   Improvement: +{improvement:.4f} (tuned better)")
                else:
                    decline = original_score - tuned_score
                    logger.info(f"   Decline: -{decline:.4f} (original better)")
            
            logger.info("🏆" + "="*60)
            
            # Copy the selected submission file as submission.csv
            target_submission = target_path / "submission.csv"
            shutil.copy2(best_submission_file, target_submission)
            logger.info(f"✅ Copied best submission from {best_submission_file} to {target_submission}")
            files_copied.append("submission.csv")
            
            # Also copy the final executable code for reference
            source_code = source_path / "states" / "final_executable_code.py"
            if source_code.exists():
                target_code = target_path / "final_executable_code.py"
                shutil.copy2(source_code, target_code)
                logger.info(f"✅ Copied final code to {target_code}")
                files_copied.append("final_executable_code.py")
            
            # Copy hyperparameter tuning results if they exist (regardless of which submission was selected)
            hyperparam_results = source_path / "hyperparam_results.json"
            if hyperparam_results.exists():
                target_hyperparam = target_path / "hyperparam_results.json"
                shutil.copy2(hyperparam_results, target_hyperparam)
                logger.info(f"✅ Copied hyperparameter results to {target_hyperparam}")
                files_copied.append("hyperparam_results.json")
            
            # Copy comparison metadata with performance information
            metadata = {
                "source_iteration": source_path.name,
                "submission_type": submission_type,
                "submission_identifier": best_submission_identifier,
                "source_file": best_submission_file.name,
                "selection_reason": reason,
                "performance_comparison": decision_info,
                "copied_at": datetime.now().isoformat(),
                "files_copied": files_copied
            }
            metadata_file = target_path / "selection_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Selection metadata saved to {metadata_file}")
            return True
                
        except Exception as e:
            logger.error(f"❌ Error copying best submission: {e}")
            return False

    def run_pipeline_single_iteration(self, iteration_type: str):
        """Run the pipeline with a single specific iteration approach."""
        logger.info(f"Starting Single-Iteration AutoML Pipeline ({iteration_type})...")
        
        # Run shared analysis steps once
        logger.info("Running shared analysis steps...")
        success = self._run_shared_analysis()
        if not success:
            return
        
        # Create iteration-specific output folder
        # For single iteration, use simple naming without numbers
        iteration_info = {
            "traditional": {"folder": "iteration_traditional", "description": "Traditional ML algorithms"},
            "custom_nn": {"folder": "iteration_custom_nn", "description": "Custom Neural Networks"}, 
            "pretrained": {"folder": "iteration_pretrained", "description": "Pretrained Models"}
        }
        
        info = iteration_info.get(iteration_type, {"folder": f"iteration_{iteration_type}", "description": iteration_type})
        iteration_output = os.path.join(self.output_folder, info['folder'])
        os.makedirs(iteration_output, exist_ok=True)
        
        # Store original and temporarily change output folder
        original_output_folder = self.output_folder
        self.output_folder = iteration_output
        
        logger.info(f"=== Running {info['description']} ===")
        
        # Run iteration-specific pipeline
        success = self._run_iteration_pipeline(iteration_type)
        if success:
            logger.info(f"Single-iteration pipeline ({iteration_type}) completed successfully!")
        else:
            logger.error(f"Single-iteration pipeline ({iteration_type}) failed!")
        
        # Restore original output folder
        self.output_folder = original_output_folder
    
    def _run_shared_analysis(self):
        """Run the shared analysis steps (description, profiling, summarization, model retrieval)."""
        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return False
        logger.info(f"Analysis result: {analysis_result}")
        self.description_analysis = analysis_result

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return False
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        # Step 3a: Summarize profiling via LLM to reduce noise
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return False
        self.profiling_summary = profiling_summary

        # Step 3b: Retrieve pretrained model/embedding suggestions
        model_suggestions = self.model_retriever_agent()
        self.model_suggestions = model_suggestions
        
        return True
    
    def _run_iteration_pipeline(self, iteration_type):
        """Run the pipeline for a specific iteration type."""
        # Step 1: Run guideline agent with iteration-specific algorithm constraint
        guideline = self.guideline_agent(iteration_type=iteration_type)
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return False
        self.guideline = guideline
        logger.info(f"Guideline generated successfully for {iteration_type}.")

        # Step 2: Run Preprocessing Coder Agent
        preprocessing_code_result = self.preprocessing_coder_agent(iteration_type=iteration_type)
        if preprocessing_code_result.get("status") == "failed":
            logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
            return False
        self.preprocessing_code = preprocessing_code_result.get("code")
        logger.info("Preprocessing code generated and validated successfully.")

        # Step 3: Run Modeling Coder Agent
        modeling_code_result = self.modeling_coder_agent(iteration_type=iteration_type)
        if modeling_code_result.get("status") == "failed":
            logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
            return False
        self.modeling_code = modeling_code_result.get("code")
        logger.info("Modeling code generated successfully.")
        # Step 4: Run Assembler Agent
        assembler_result = self.assembler_agent(iteration_type=iteration_type)
        if assembler_result.get("status") == "failed":
            logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}")
            return False
        self.assembled_code = assembler_result.get("code")
        logger.info("Final script generated and executed successfully.")
        # Step 5: Hyperparameter tuning for all iterations (traditional, custom_nn, pretrained)
        # 🚨 CRITICAL: This step MUST generate submission_tuned.csv, NOT submission.csv 🚨
        if iteration_type in ("traditional", "custom_nn", "pretrained"):
            logger.info(f"🚨 STARTING HYPERPARAMETER TUNING - MUST GENERATE submission_tuned.csv 🚨")
            hpt_result = self.hyperparameter_tuning_agent(iteration_type=iteration_type)
            if hpt_result.get("status") == "failed":
                logger.warning(f"Hyperparameter tuning failed for {iteration_type}: {hpt_result.get('error')}.")
                logger.info("Proceeding with submission from last assembled code without tuning.")
                # Keep original submission file as the final result
                self.final_submission_path = assembler_result.get("submission_path")
            else:
                self.hyperparameter_tuning_results = hpt_result.get("results")
                logger.info(f"🎉 Hyperparameter tuning completed for {iteration_type} 🎉")
                # Check if tuned submission file exists in the current iteration folder
                # self.output_folder is already the iteration folder (e.g., /output/.../iteration_2_custom_nn)
                tuned_submission_path = os.path.join(self.output_folder, "submission_tuned.csv")
                if os.path.exists(tuned_submission_path):
                    self.final_submission_path = tuned_submission_path
                    logger.info(f"✅ ✅ CORRECT: Using tuned submission file: {tuned_submission_path} ✅ ✅")
                else:
                    self.final_submission_path = assembler_result.get("submission_path")
                    logger.error(f"🚨 ERROR: submission_tuned.csv not found! Using original: {self.final_submission_path}")
                    logger.error(f"🚨 Hyperparameter tuning should have created submission_tuned.csv! 🚨")
        else:
            # For non-tuning iterations, use original submission
            self.final_submission_path = assembler_result.get("submission_path")
        
        # Performance summary for this iteration
        self._log_iteration_performance_summary(iteration_type, assembler_result, hpt_result if 'hpt_result' in locals() else None)
        
        return True

    def _log_iteration_performance_summary(self, iteration_type: str, assembler_result: Dict, hpt_result: Optional[Dict] = None):
        """Log performance summary for an iteration."""
        iteration_name = os.path.basename(self.output_folder)
        
        logger.info("📊" + "="*60)
        logger.info(f"🎯 ITERATION PERFORMANCE SUMMARY: {iteration_name}")
        logger.info("📊" + "="*60)
        
        # Try to load scores from separate JSON files
        non_tuned_file = os.path.join(self.output_folder, "non_tuned_scores.json")
        tuned_file = os.path.join(self.output_folder, "tuned_scores.json")
        
        # Log original scores
        if os.path.exists(non_tuned_file):
            try:
                with open(non_tuned_file, 'r', encoding='utf-8') as f:
                    non_tuned_data = json.load(f)
                
                original_scores = non_tuned_data.get("scores", {})
                if original_scores:
                    logger.info(f"📈 Original submission metrics: {original_scores}")
                    # Get primary score
                    primary_score = None
                    score_priority = ['validation_score', 'accuracy', 'f1_score', 'auc', 'r2_score']
                    for score_name in score_priority:
                        if score_name in original_scores:
                            primary_score = original_scores[score_name]
                            logger.info(f"🏆 Original primary score: {primary_score:.4f} ({score_name})")
                            break
                    if primary_score is None:
                        logger.warning("⚠️ No primary score identified from original metrics")
                else:
                    logger.warning("⚠️ No original scores found in non_tuned_scores.json")
                    
            except Exception as e:
                logger.warning(f"Failed to read non_tuned_scores.json: {e}")
        else:
            logger.warning("⚠️ non_tuned_scores.json not found")
        
        # Log tuned scores if available
        if os.path.exists(tuned_file):
            try:
                with open(tuned_file, 'r', encoding='utf-8') as f:
                    tuned_data = json.load(f)
                
                tuned_scores = tuned_data.get("scores", {})
                best_tuned = tuned_data.get("best_score")
                if tuned_scores:
                    logger.info(f"📈 Tuned submission metrics: {tuned_scores}")
                    if best_tuned is not None:
                        logger.info(f"🏆 Best tuned score: {best_tuned:.4f}")
                    else:
                        logger.warning("⚠️ No best tuned score identified")
                else:
                    logger.warning("⚠️ No tuned scores found in tuned_scores.json")
                    
            except Exception as e:
                logger.warning(f"Failed to read tuned_scores.json: {e}")
        
        # Log file locations for easy comparison
        if os.path.exists(non_tuned_file) or os.path.exists(tuned_file):
            logger.info(f"📋 Score files for comparison:")
            if os.path.exists(non_tuned_file):
                logger.info(f"   📄 Original: {non_tuned_file}")
            if os.path.exists(tuned_file):
                logger.info(f"   📄 Tuned: {tuned_file}")
        else:
            # Fallback to old method
            self._log_metrics_fallback(assembler_result, hpt_result)
        
        logger.info("📊" + "="*60)
    
    def _log_metrics_fallback(self, assembler_result: Dict, hpt_result: Optional[Dict] = None):
        """Fallback method to log metrics when scores.json is not available."""
        # Log assembler (original) performance
        assembler_metrics = assembler_result.get("scores", {})
        if assembler_metrics:
            logger.info(f"📈 Original submission metrics: {assembler_metrics}")
            # Get primary score
            primary_score = None
            score_priority = ['validation_score', 'accuracy', 'f1_score', 'auc', 'r2_score']
            for score_name in score_priority:
                if score_name in assembler_metrics:
                    primary_score = assembler_metrics[score_name]
                    logger.info(f"🏆 Original primary score: {primary_score:.4f} ({score_name})")
                    break
            if primary_score is None:
                logger.warning("⚠️ No primary score identified from original metrics")
        else:
            logger.warning("⚠️ No original performance metrics available")
        
        # Log hyperparameter tuning performance if available
        if hpt_result and hpt_result.get("status") == "success":
            hpt_metrics = hpt_result.get("metrics", {})
            if hpt_metrics:
                logger.info(f"🎯 Tuned submission metrics: {hpt_metrics}")
                # Get primary tuning score
                primary_tuning_score = None
                tuning_score_priority = ['best_score', 'best_params_score', 'final_score', 'tuned_score', 'validation_score', 'accuracy', 'f1_score']
                for score_name in tuning_score_priority:
                    if score_name in hpt_metrics:
                        primary_tuning_score = hpt_metrics[score_name]
                        logger.info(f"🏆 Best tuned score: {primary_tuning_score:.4f} ({score_name})")
                        break
                if primary_tuning_score is None:
                    logger.warning("⚠️ No primary score identified from tuning metrics")
            else:
                logger.warning("⚠️ No tuning performance metrics available")
        else:
            logger.warning("⚠️ No tuned performance metrics available")
        
        if hpt_result and hpt_result.get("status") == "failed":
            logger.warning("⚠️ Hyperparameter tuning failed - using original submission")
        elif hpt_result is None:
            logger.info("ℹ️ No hyperparameter tuning performed for this iteration")
        
        # Check which submission files exist
        original_exists = os.path.exists(os.path.join(self.output_folder, "submission.csv"))
        tuned_exists = os.path.exists(os.path.join(self.output_folder, "submission_tuned.csv"))
        
        logger.info(f"📁 Generated files:")
        if original_exists:
            logger.info(f"   ✅ submission.csv (original)")
        if tuned_exists:
            logger.info(f"   ✅ submission_tuned.csv (hyperparameter tuned)")
        
        logger.info("📊" + "="*60)

    def run_pipeline(self):
        """Run the entire pipeline from description analysis to code generation."""

        # Step 1: Run description analysis agent
        analysis_result = self.description_analyzer_agent()
        if "error" in analysis_result:
            logger.error(f"Description analysis failed: {analysis_result['error']}")
            return
        logger.info(f"Analysis result: {analysis_result}")

        self.description_analysis = analysis_result

        # Step 2: Run profiling agent
        profiling_result = self.profiling_agent()
        if "error" in profiling_result:
            logger.error(f"Data profiling failed: {profiling_result['error']}")
            return
        
        self.profiling_result = profiling_result
        logger.info("Profiling overview generated.")

        # Step 3: Run guideline agent
        # 3a: Summarize profiling via LLM to reduce noise
        profiling_summary = self.profiling_summarizer_agent()
        if "error" in profiling_summary:
            logger.error(f"Profiling summarization failed: {profiling_summary['error']}")
            return
        self.profiling_summary = profiling_summary

        # 3b: Retrieve pretrained model/embedding suggestions
        model_suggestions = self.model_retriever_agent()
        self.model_suggestions = model_suggestions

        # 3c: Run guideline agent with summarized profiling + model suggestions
        guideline = self.guideline_agent()
        if "error" in guideline:
            logger.error(f"Guideline generation failed: {guideline['error']}")
            return
        
        self.guideline = guideline
        logger.info("Guideline generated successfully.")

        # Step 4: Run Preprocessing Coder Agent
        preprocessing_code_result = self.preprocessing_coder_agent()
        if preprocessing_code_result.get("status") == "failed":
            logger.error(f"Preprocessing code generation failed: {preprocessing_code_result.get('error')}")
            return

        self.preprocessing_code = preprocessing_code_result.get("code")
        logger.info("Preprocessing code generated and validated successfully.")

        # Step 5: Run Modeling Coder Agent
        modeling_code_result = self.modeling_coder_agent()
        if modeling_code_result.get("status") == "failed":
            logger.error(f"Modeling code generation failed: {modeling_code_result.get('error')}")
            return
            
        self.modeling_code = modeling_code_result.get("code")
        logger.info("Modeling code generated successfully (not yet validated).")

        # Step 6: Run Assembler Agent to assemble, finalize, and run the code
        assembler_result = self.assembler_agent()
        if assembler_result.get("status") == "failed":
            logger.error(f"Final code assembly and execution failed: {assembler_result.get('error')}" )
            return
        self.assembled_code = assembler_result.get("code")
        logger.info(f"Initial script generated and executed successfully.")
        # Step 7: Run hyperparameter tuning phase on assembled code
        # 🚨 CRITICAL: This step MUST generate submission_tuned.csv, NOT submission.csv 🚨
        logger.info(f"🚨 STARTING HYPERPARAMETER TUNING - MUST GENERATE submission_tuned.csv 🚨")
        hpt_result = self.hyperparameter_tuning_agent()
        if hpt_result.get("status") == "failed":
            logger.error(f"Hyperparameter tuning failed: {hpt_result.get('error')}")
            logger.error(f"🚨 FAILED TO GENERATE submission_tuned.csv! 🚨")
            return
        self.hyperparameter_tuning_results = hpt_result.get("results")
        logger.info("🎉 Hyperparameter tuning completed successfully - submission_tuned.csv should be generated! 🎉")

        logger.info("AutoML pipeline completed successfully!")

    def write_code_script(self, script, output_code_file):
        with open(output_code_file, "w") as file:
            file.write(script)

    def execute_code(self, code_to_execute: str, phase_name: str, attempt: int) -> dict:
        """
        Executes a string of Python code in a subprocess and saves the script,
        stdout, and stderr to a structured attempts folder.

        Args:
            code_to_execute: The Python code to run.
            phase_name: The name of the phase (e.g., "preprocessing", "assembler").
            attempt: The retry attempt number.

        Returns:
            A dictionary with execution status, stdout, and stderr.
        """
        # Create a structured directory for this attempt
        attempt_dir = Path(self.output_folder) / "attempts" / phase_name / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths for the script, stdout, and stderr
        script_path = attempt_dir / "code_generated.py"
        stdout_path = attempt_dir / "stdout.txt"
        stderr_path = attempt_dir / "stderr.txt"

        # Write the code to the script file
        self.write_code_script(code_to_execute, str(script_path))

        logger.info(f"Executing code from: {script_path}")

        try:
            # Execute the script using subprocess
            working_dir = str(Path(self.input_data_folder).parent)
            
            process = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                check=False,  # Do not raise exception on non-zero exit code
                cwd=working_dir,
                timeout=self.config.per_execution_timeout,
            )

            stdout = process.stdout
            stderr = process.stderr

            # Save stdout and stderr to their respective files
            with open(stdout_path, "w") as f:
                f.write(stdout)
            with open(stderr_path, "w") as f:
                f.write(stderr)

            if process.returncode == 0:
                logger.info("Code executed successfully.")
                return {"success": True, "stdout": stdout, "stderr": stderr}
            else:
                logger.error(f"Code execution failed with return code {process.returncode}.")
                full_error = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                return {"success": False, "stdout": stdout, "stderr": full_error}

        except subprocess.TimeoutExpired as e:
            logger.error(f"Code execution timed out after {self.config.per_execution_timeout} seconds.")
            full_error = f"Timeout Error: Execution exceeded {self.config.per_execution_timeout} seconds.\n\nSTDOUT:\n{e.stdout or ''}\n\nSTDERR:\n{e.stderr or ''}"
            # Save partial output if available
            with open(stdout_path, "w") as f:
                f.write(e.stdout or "")
            with open(stderr_path, "w") as f:
                f.write(e.stderr or "")
            return {"success": False, "stdout": e.stdout, "stderr": full_error}
        except Exception as e:
            logger.error(f"An exception occurred during code execution: {e}")
            # Save exception to stderr file
            with open(stderr_path, "w") as f:
                f.write(str(e))
            return {"success": False, "stdout": "", "stderr": str(e)}


    def update_python_code(self):
        """Update the current Python code."""
        assert len(self.python_codes) == self.time_step
        assert len(self.python_file_paths) == self.time_step

        python_code = self.python_coder()

        python_file_path = os.path.join(self.iteration_folder, "generated_code.py")

        self.write_code_script(python_code, python_file_path)

        self.python_codes.append(python_code)
        self.python_file_paths.append(python_file_path)

    def update_bash_script(self):
        """Update the current bash script."""
        assert len(self.bash_scripts) == self.time_step

        bash_script = self.bash_coder()

        bash_file_path = os.path.join(self.iteration_folder, "execution_script.sh")

        self.write_code_script(bash_script, bash_file_path)

        self.bash_scripts.append(bash_script)

    def execute_code_old(self):
        planner_decision, planner_error_summary, planner_prompt, stderr, stdout = self.executer(
            code_to_execute=self.bash_script,
            code_to_analyze=self.python_code,
            task_description=self.task_description,
            data_prompt=self.data_prompt,
        )

        self.save_and_log_states(stderr, "stderr", add_uuid=False)
        self.save_and_log_states(stdout, "stdout", add_uuid=False)

        if planner_decision == "FIX":
            logger.brief(f"[bold red]Code generation failed in iteration[/bold red] {self.time_step}!")
            # Add suggestions to the error message to guide next iteration
            error_message = f"stderr: {stderr}\n\n" if stderr else ""
            error_message += (
                f"Error summary from planner (the error can appear in stdout if it's catched): {planner_error_summary}"
            )
            self.update_error_message(error_message=error_message)
            return False
        elif planner_decision == "FINISH":
            logger.brief(
                f"[bold green]Code generation successful after[/bold green] {self.time_step + 1} [bold green]iterations[/bold green]"
            )
            self.update_error_message(error_message="")
            return True
        else:
            logger.warning(f"###INVALID Planner Output: {planner_decision}###")
            self.update_error_message(error_message="")
            return False

    def update_error_message(self, error_message: str):
        """Update the current error message."""
        assert len(self.error_messages) == self.time_step
        self.error_messages.append(error_message)

    def save_and_log_states(self, content, save_name, add_uuid=False):
        if add_uuid:
            # Split filename and extension
            name, ext = os.path.splitext(save_name)
            # Generate 4-digit UUID (using first 4 characters of hex)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        states_dir = os.path.join(self.output_folder, "states")
        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is not None:
                if isinstance(content, list):
                    # Join list elements with newlines
                    file.write("\n".join(str(item) for item in content))
                else:
                    # Handle as string (original behavior)
                    file.write(content)
            else:
                file.write("<None>")

    def log_agent_start(self, message: str):
        logger.brief(message)

    def log_agent_end(self, message: str):
        logger.brief(message)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
