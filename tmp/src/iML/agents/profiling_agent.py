import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm
from ydata_profiling import ProfileReport

from .base_agent import BaseAgent

# Configure logger
logger = logging.getLogger(__name__)

class ProfilingAgent(BaseAgent):
    """
    This agent performs data profiling based on analysis results
    from DescriptionAnalyzerAgent. It reads CSV files, creates reports,
    and returns a comprehensive summary.
    """

    def __init__(self, config, manager):
        super().__init__(config=config, manager=manager)
        # This agent doesn't need LLM or specific prompt configuration
        logger.info("ProfilingAgent initialized.")

    def __call__(self) -> Dict[str, Any]:
        """
        Main execution method of the agent.
        """
        self.manager.log_agent_start("ProfilingAgent: Starting Data Profiling...")

        # Get analysis results from manager
        description_analysis = self.manager.description_analysis
        if not description_analysis or "error" in description_analysis:
            logger.error("ProfilingAgent: description_analysis is missing or contains an error. Skipping.")
            return {"error": "Input description_analysis not available."}

        ds_name = description_analysis.get('name', 'unnamed_dataset')
        logger.info(f"Profiling dataset: {ds_name}")

        # Check existence of data files
        paths_list = description_analysis.get("link to the dataset", [])
        if not isinstance(paths_list, list):
            logger.warning("'link to the dataset' is not a list. Skipping profiling.")
            paths_list = []

        path_status = self._check_paths(paths_list)
        
        # Filter and profile existing CSV files
        existing_csv_paths = [p for p in path_status["exists"] if p.lower().endswith(".csv")]

        # --- CHANGE: Save profiling results to memory ---
        all_summaries = {}
        all_profiles = {}

        for p_str in tqdm(existing_csv_paths, desc="Profiling CSV files"):
            csv_path = Path(p_str)
            file_stem = csv_path.stem
            summary, profile_content = self._profile_csv(csv_path)
            
            # Only add if profiling is successful
            if summary:
                all_summaries[file_stem] = summary
            if profile_content:
                all_profiles[file_stem] = profile_content
            
        # Combine back into a single object
        profiling_result = {
            "summaries": all_summaries,
            "profiles": all_profiles
        }

        # Save aggregated results to the run's states directory
        self.manager.save_and_log_states(
            content=json.dumps(profiling_result, indent=2, ensure_ascii=False),
            save_name="profiling_result.json" # File name as requested
        )

        self.manager.log_agent_end(f"ProfilingAgent: Profiling COMPLETED.")
        
        # Return aggregated results
        return profiling_result

    def _check_paths(self, paths: List[str]) -> Dict[str, List[str]]:
        """Check if file paths exist."""
        exists, missing = [], []
        for p in paths:
            pth = Path(p)
            (exists if pth.exists() else missing).append(p)
        return {"exists": exists, "missing": missing}

    def _filter_value_counts(self, profile_json_str: str) -> str:
        """Remove heavy parts from profile JSON to reduce file size."""
        try:
            profile_dict = json.loads(profile_json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error while filtering profile: {e}")
            return profile_json_str
        
        if "variables" not in profile_dict:
            return profile_json_str
            
        for var_info in profile_dict.get("variables", {}).values():
            var_type = var_info.get("type", "")
            n_unique = var_info.get("n_unique", 0)
            should_remove = (
                var_type in ["Text", "Numeric", "Date", "DateTime", "Time", "URL", "Path"]
                or (var_type == "Categorical" and n_unique > 50)
            )
            if should_remove:
                keys_to_remove = [
                    "value_counts_without_nan", "value_counts_index_sorted", "histogram",
                    "length_histogram", "histogram_length", "block_alias_char_counts",
                    "word_counts", "category_alias_char_counts", "script_char_counts",
                    "block_alias_values", "category_alias_values", "character_counts",
                    "block_alias_counts", "script_counts", "category_alias_counts",
                    "n_block_alias", "n_scripts", "n_category",
                ]
                for key in keys_to_remove:
                    var_info.pop(key, None)
        return json.dumps(profile_dict, ensure_ascii=False, indent=2)

    def _profile_csv(self, csv_path: Path) -> tuple[Any, Any]:
        """Create profile report for a CSV file and return (summary, profile_content)."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Analyzing {csv_path.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            profile = ProfileReport(
                df,
                title=f"Profile - {csv_path.name}",
                minimal=True,
                samples={"random": 5},
                correlations={"auto": {"calculate": False}},
                missing_diagrams={"bar": False, "matrix": False},
                interactions={"targets": []},
                explorative=False,
                progress_bar=False,
                infer_dtypes=True
            )
            
            filtered_json_str = self._filter_value_counts(profile.to_json())
            profile_content = json.loads(filtered_json_str)

            summary = {
                "file": str(csv_path),
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_pct": df.isnull().mean().round(4).to_dict(),
                "file_size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2)
            }
            
            logger.info(f"In-memory profile created for {csv_path.name}")
            return summary, profile_content
        except Exception as e:
            logger.error(f"Failed to profile {csv_path.name}: {e}")
            error_summary = {"error": str(e), "file": str(csv_path)}
            return error_summary, None
