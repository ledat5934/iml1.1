import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from iML.main_runner import run_automl_pipeline

def main():
    """
    Main entry point for the application when run from the terminal.
    Parses input arguments and calls the main pipeline.
    """
    parser = argparse.ArgumentParser(description="iML")
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input data folder"
    )
    parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Path to the output directory. If not provided, one will be auto-generated in the 'runs/' directory."
    )
    parser.add_argument(
        "-c", "--config", 
        default="configs/default.yaml", 
        help="Path to the configuration file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=["full", "partial", "resume"],
        default="full",
        help="Pipeline execution mode: 'full' (complete run), 'partial' (stop at checkpoint), 'resume' (continue from checkpoint)"
    )
    parser.add_argument(
        "--checkpoint-action",
        default="guideline",
        help="For partial mode: where to stop ('description', 'profiling', 'guideline'). For resume mode: where to start ('preprocessing', 'modeling', 'assembler')"
    )
    
    args = parser.parse_args()
    
    # Call the main pipeline function from main_runner
    run_automl_pipeline(
        input_data_folder=args.input,
        output_folder=args.output,
        config_path=args.config,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_action=args.checkpoint_action,
    )

if __name__ == "__main__":
    main()