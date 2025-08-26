import os
import pandas as pd
from pathlib import Path
from typing import List

def _build_tree_conditional_limit(
    dir_path: Path,
    prefix: str,
    lines: List[str],
    csv_paths_collector: List[Path],
    root_dir: Path,
):
    """
    Recursive helper function to build the directory tree with conditional file limiting.
    - If a directory has > 50 files, it shows the first 3 and an ellipsis (...).
    - Otherwise, it shows all files.
    """
    try:
        # Get all entries and sort them with directories first
        entries = sorted(
            [p for p in dir_path.iterdir()],
            key=lambda p: (p.is_file(), p.name.lower())
        )
    except OSError as e:
        lines.append(f"{prefix}└── [Error reading directory: {e}]")
        return

    # Separate directories and all files
    dirs = [e for e in entries if e.is_dir()]
    all_files = [e for e in entries if e.is_file() and e.name.lower() != "description.txt"]

    # --- START of New Conditional Logic ---
    display_files = []
    has_more_files = False

    if len(all_files) > 50:
        display_files = all_files[:3]
        has_more_files = True
    else:
        display_files = all_files
        has_more_files = False
    # --- END of New Conditional Logic ---

    # Combine directories and the limited list of files for display in the tree
    entries_to_process = dirs + display_files

    # Process each entry to build the tree structure
    for i, entry in enumerate(entries_to_process):
        # Determine if this entry is the last one to be displayed
        is_last_in_list = (i == len(entries_to_process) - 1)
        is_last_node = is_last_in_list and not has_more_files

        connector = "└── " if is_last_node else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            # If it's a directory, continue recursively with the correct prefix
            new_prefix = prefix + ("    " if is_last_node else "│   ")
            _build_tree_conditional_limit(
                entry, new_prefix, lines, csv_paths_collector, root_dir
            )

    # If files were omitted based on the condition, add an ellipsis "..." indicator
    if has_more_files:
        lines.append(f"{prefix}└── ...")

    # Go through ALL files (not just the displayed ones) to collect every CSV
    for file_entry in all_files:
        if file_entry.suffix.lower() == ".csv":
            rel_path = file_entry.relative_to(root_dir)
            if rel_path not in csv_paths_collector:
                csv_paths_collector.append(rel_path)


def get_directory_structure(
    root_dir: str,
    sample_rows: int = 5
) -> str:
    """
    Generates a string representing the directory structure as a tree 
    and provides a summary of all CSV files found.

    The tree display has a conditional rule:
    - If a directory contains more than 50 files, only the first 3 are shown,
      followed by "...".
    - If it contains 50 or fewer files, all files are shown.

    Args:
        root_dir (str): Path to the root directory.
        sample_rows (int): Number of sample rows to display from each CSV file summary.

    Returns:
        str: A string containing the directory tree and CSV summary.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"'{root_dir}' is not a valid directory.")

    # --- Part 1: Build the directory tree ---
    tree_lines: List[str] = [f"{root_path.name}/"]
    csv_paths: List[Path] = []
    
    # Call the recursive helper with the new conditional logic
    _build_tree_conditional_limit(
        dir_path=root_path,
        prefix="",
        lines=tree_lines,
        csv_paths_collector=csv_paths,
        root_dir=root_path
    )

    summary_lines: List[str] = []
    # --- Part 2: Summarize the collected CSV files ---
    if csv_paths:
        summary_lines.append("\n" + "="*60)
        summary_lines.append("SUMMARY OF ALL CSV FILES")
        summary_lines.append("="*60)

        csv_paths.sort()
        for rel_path in csv_paths:
            abs_path = root_path / rel_path
            try:
                df = pd.read_csv(abs_path, nrows=sample_rows, on_bad_lines='skip')
            except Exception as e:
                summary_lines.append(f"\nCould not read file '{rel_path}': {e}")
                continue
            
            summary_lines.append(f"\nStructure of file: {rel_path}")
            summary_lines.append("   Columns: " + ", ".join(df.columns.astype(str)))
            summary_lines.append("   First few rows:")
            df_string = df.to_string(index=False)
            indented_df_string = "   " + df_string.replace("\n", "\n   ")
            summary_lines.append(indented_df_string)

    # --- Part 3: Combine and return the final string ---
    final_output = "\n".join(tree_lines) + "\n".join(summary_lines)
    return final_output