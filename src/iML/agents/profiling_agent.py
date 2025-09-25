import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
            
        # Add ID format analysis
        id_format_analysis = self._analyze_id_formats(existing_csv_paths)
        
        # Combine back into a single object
        profiling_result = {
            "summaries": all_summaries,
            "profiles": all_profiles,
            "id_format_analysis": id_format_analysis
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

    def _analyze_id_formats(self, csv_paths: List[str]) -> Dict[str, Any]:
        """Analyze ID column formats across CSV files to detect file extensions."""
        logger.info("Analyzing ID formats for file extensions...")
        
        analysis_result = {
            "has_file_extensions": False,
            "detected_extensions": [],
            "id_columns_info": {},
            "submission_format_analysis": None,
            "format_notes": []
        }
        
        # Common file extensions to look for
        common_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.csv', '.txt', '.pdf', '.mp4', '.avi', '.wav', '.mp3']
        extension_pattern = r'\.(jpg|jpeg|png|gif|bmp|tiff|csv|txt|pdf|mp4|avi|wav|mp3)$'
        
        detected_extensions = set()
        
        for csv_path_str in csv_paths:
            try:
                csv_path = Path(csv_path_str)
                df = pd.read_csv(csv_path)
                
                # Identify potential ID columns
                id_columns = self._identify_id_columns(df)
                
                for col_name in id_columns:
                    col_info = self._analyze_id_column(df[col_name], col_name, extension_pattern)
                    
                    if col_info["has_extensions"]:
                        analysis_result["has_file_extensions"] = True
                        detected_extensions.update(col_info["extensions_found"])
                        analysis_result["format_notes"].append(
                            f"Column '{col_name}' in {csv_path.name} contains file extensions: {col_info['extensions_found']}"
                        )
                    
                    analysis_result["id_columns_info"][f"{csv_path.stem}_{col_name}"] = col_info
                    
            except Exception as e:
                logger.warning(f"Failed to analyze ID formats in {csv_path_str}: {e}")
        
        analysis_result["detected_extensions"] = sorted(list(detected_extensions))
        
        # Analyze submission format if available
        submission_analysis = self._analyze_submission_format()
        if submission_analysis:
            analysis_result["submission_format_analysis"] = submission_analysis
            
            # Compare formats and add notes
            if analysis_result["has_file_extensions"] and submission_analysis.get("submission_has_extensions") is False:
                analysis_result["format_notes"].append(
                    "IMPORTANT: Training data IDs contain file extensions but submission format appears to need IDs without extensions"
                )
            elif not analysis_result["has_file_extensions"] and submission_analysis.get("submission_has_extensions") is True:
                analysis_result["format_notes"].append(
                    "IMPORTANT: Training data IDs do not contain extensions but submission format appears to need IDs with extensions"
                )
            elif analysis_result["has_file_extensions"] and submission_analysis.get("submission_has_extensions") is True:
                analysis_result["format_notes"].append(
                    "IMPORTANT: Training data IDs contain file extensions and submission format appears to need IDs with extensions"
                )
            elif not analysis_result["has_file_extensions"] and submission_analysis.get("submission_has_extensions") is False:
                analysis_result["format_notes"].append(
                    "IMPORTANT: Training data IDs do not contain extensions and submission format appears to need IDs without extensions"
                )
        
        logger.info(f"ID format analysis completed. Found extensions: {analysis_result['detected_extensions']}")
        return analysis_result

    def _identify_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify potential ID columns in the dataframe."""
        id_columns = []
        
        # Common ID column names
        id_patterns = ['id', 'ID', 'Id', 'image_id', 'file_id', 'filename', 'file_name', 'image_name']
        
        for col in df.columns:
            # Check exact matches
            if col in id_patterns:
                id_columns.append(col)
                continue
                
            # Check partial matches
            col_lower = col.lower()
            if any(pattern.lower() in col_lower for pattern in ['id', 'filename', 'file_name', 'image']):
                # Additional check: should have mostly unique values
                if df[col].nunique() / len(df) > 0.8:  # More than 80% unique
                    id_columns.append(col)
        
        # If no ID columns found by name, look for first column with high uniqueness
        if not id_columns:
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() / len(df) > 0.9:
                    id_columns.append(col)
                    break
        
        return id_columns

    def _analyze_id_column(self, series: pd.Series, col_name: str, extension_pattern: str) -> Dict[str, Any]:
        """Analyze a specific ID column for file extensions."""
        sample_values = series.dropna().head(100).astype(str)
        
        # Find values with extensions
        values_with_extensions = []
        extensions_found = set()
        
        for value in sample_values:
            match = re.search(extension_pattern, value, re.IGNORECASE)
            if match:
                values_with_extensions.append(value)
                extensions_found.add('.' + match.group(1).lower())
        
        has_extensions = len(values_with_extensions) > 0
        extension_ratio = len(values_with_extensions) / len(sample_values) if sample_values.any() else 0
        
        return {
            "column_name": col_name,
            "has_extensions": has_extensions,
            "extensions_found": sorted(list(extensions_found)),
            "extension_ratio": round(extension_ratio, 3),
            "sample_with_extensions": values_with_extensions[:5],
            "total_samples_checked": len(sample_values)
        }

    def _analyze_submission_format(self) -> Optional[Dict[str, Any]]:
        """Analyze sample submission file format if available."""
        # Look for common submission file names in input directory
        input_path = Path(self.manager.input_data_folder)
        submission_patterns = ['sample_submission.csv', 'submission.csv', 'sample_submit.csv', 'submit.csv']
        
        for pattern in submission_patterns:
            submission_file = input_path / pattern
            if submission_file.exists():
                try:
                    logger.info(f"Found submission file: {submission_file.name}")
                    df = pd.read_csv(submission_file)
                    
                    # Analyze first few rows
                    if len(df) > 0 and len(df.columns) > 0:
                        first_col = df.columns[0]  # Assume first column is ID
                        sample_ids = df[first_col].head(10).astype(str)
                        
                        # Check for extensions
                        extension_pattern = r'\.(jpg|jpeg|png|gif|bmp|tiff|csv|txt|pdf|mp4|avi|wav|mp3)$'
                        has_extensions = any(re.search(extension_pattern, str(id_val), re.IGNORECASE) for id_val in sample_ids)
                        
                        return {
                            "submission_file": str(submission_file),
                            "submission_has_extensions": has_extensions,
                            "first_column_name": first_col,
                            "sample_ids": sample_ids.tolist()[:3],
                            "total_rows": len(df)
                        }
                except Exception as e:
                    logger.warning(f"Failed to analyze submission file {submission_file}: {e}")
        
        return None
