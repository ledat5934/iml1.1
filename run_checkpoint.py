#!/usr/bin/env python3
"""
Helper script for running AutoML pipeline with checkpoint functionality.
This script provides convenient commands for running partial pipelines and resuming from checkpoints.
"""
import argparse
import sys
from pathlib import Path
import os

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from iML.main_runner import run_automl_pipeline

def main():
    """
    Enhanced entry point with checkpoint functionality examples and shortcuts.
    """
    parser = argparse.ArgumentParser(
        description="iML AutoML Pipeline with Checkpoint Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run until guideline generation (default behavior for manual editing)
  python run_checkpoint.py -i ./data --stop-at-guideline

  # Run until profiling only  
  python run_checkpoint.py -i ./data --stop-at-profiling

  # Resume from preprocessing (after manual guideline editing)
  python run_checkpoint.py -i ./data --resume-from-preprocessing -o ./runs/run_20240101_120000_abcd1234

  # Resume from modeling only
  python run_checkpoint.py -i ./data --resume-from-modeling -o ./runs/run_20240101_120000_abcd1234

  # Full pipeline (original behavior)
  python run_checkpoint.py -i ./data --full

Advanced usage:
  # Custom checkpoint control
  python run_checkpoint.py -i ./data --checkpoint-mode partial --checkpoint-action description
  python run_checkpoint.py -i ./data --checkpoint-mode resume --checkpoint-action assembler -o ./existing_run_dir
        """
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input data folder"
    )
    parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Path to the output directory. Required for resume operations. Auto-generated for new runs."
    )
    parser.add_argument(
        "-c", "--config", 
        default="configs/default.yaml", 
        help="Path to the configuration file (default: configs/default.yaml)"
    )

    # Convenient shortcuts
    shortcut_group = parser.add_mutually_exclusive_group()
    shortcut_group.add_argument(
        "--full",
        action="store_true",
        help="Run the complete pipeline (default behavior)"
    )
    shortcut_group.add_argument(
        "--stop-at-guideline",
        action="store_true",
        help="Stop after guideline generation for manual editing"
    )
    shortcut_group.add_argument(
        "--stop-at-profiling",
        action="store_true",
        help="Stop after data profiling"
    )
    shortcut_group.add_argument(
        "--stop-at-description",
        action="store_true",
        help="Stop after description analysis"
    )
    shortcut_group.add_argument(
        "--resume-from-preprocessing",
        action="store_true",
        help="Resume from preprocessing step (requires existing output directory)"
    )
    shortcut_group.add_argument(
        "--resume-from-modeling",
        action="store_true",
        help="Resume from modeling step (requires existing output directory)"
    )
    shortcut_group.add_argument(
        "--resume-from-assembler",
        action="store_true",
        help="Resume from assembler step (requires existing output directory)"
    )

    # Advanced control (overrides shortcuts)
    parser.add_argument(
        "--checkpoint-mode",
        choices=["full", "partial", "resume"],
        help="Advanced: Pipeline execution mode (overrides shortcuts)"
    )
    parser.add_argument(
        "--checkpoint-action",
        help="Advanced: Checkpoint action (overrides shortcuts)"
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint mode and action from shortcuts or advanced options
    checkpoint_mode = "full"
    checkpoint_action = "run"
    
    if args.checkpoint_mode:
        # Advanced mode specified
        checkpoint_mode = args.checkpoint_mode
        checkpoint_action = args.checkpoint_action or "guideline"
    else:
        # Use shortcuts
        if args.stop_at_guideline:
            checkpoint_mode = "partial"
            checkpoint_action = "guideline"
        elif args.stop_at_profiling:
            checkpoint_mode = "partial"
            checkpoint_action = "profiling"
        elif args.stop_at_description:
            checkpoint_mode = "partial"
            checkpoint_action = "description"
        elif args.resume_from_preprocessing:
            checkpoint_mode = "resume"
            checkpoint_action = "preprocessing"
        elif args.resume_from_modeling:
            checkpoint_mode = "resume"
            checkpoint_action = "modeling"
        elif args.resume_from_assembler:
            checkpoint_mode = "resume"
            checkpoint_action = "assembler"
        # args.full or no option = default full mode
    
    # Validate resume mode requirements
    if checkpoint_mode == "resume" and args.output is None:
        print("Error: Resume mode requires an existing output directory (-o/--output)")
        print("Specify the path to the run directory containing previous checkpoint data.")
        sys.exit(1)
    
    if checkpoint_mode == "resume" and not Path(args.output).exists():
        print(f"Error: Output directory does not exist: {args.output}")
        print("For resume mode, you must specify an existing run directory.")
        sys.exit(1)

    # Print execution plan
    if checkpoint_mode == "partial":
        print(f"🚀 Running pipeline until: {checkpoint_action}")
        print("💡 After completion, you can manually edit the guideline and resume with:")
        print(f"   python run_checkpoint.py -i {args.input} --resume-from-preprocessing -o <output_dir>")
    elif checkpoint_mode == "resume":
        print(f"🔄 Resuming pipeline from: {checkpoint_action}")
        print(f"📁 Using existing run directory: {args.output}")
    else:
        print("🚀 Running complete pipeline")
    
    print()
    
    # Call the main pipeline function
    try:
        run_automl_pipeline(
            input_data_folder=args.input,
            output_folder=args.output,
            config_path=args.config,
            checkpoint_mode=checkpoint_mode,
            checkpoint_action=checkpoint_action,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
