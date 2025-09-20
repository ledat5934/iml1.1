#!/usr/bin/env python3
"""
Simplified runner script for Multi-Iteration AutoML
This script provides an easy-to-use interface for running the multi-iteration pipeline.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from iML.main_runner import run_automl_pipeline

def main():
    """
    Simplified entry point for Multi-Iteration AutoML.
    """
    parser = argparse.ArgumentParser(
        description="iML Multi-Iteration AutoML - Generate and compare 3 different ML solutions",
        epilog="""
Examples:
  # Run all 3 iterations and get best solution
  python run_multi_iteration.py -i ./my_dataset
  
  # Run with specific config
  python run_multi_iteration.py -i ./my_dataset -c configs/openai_config.yaml
  
  # Run single iteration type
  python run_multi_iteration.py -i ./my_dataset --single traditional
  
  # Custom output folder
  python run_multi_iteration.py -i ./my_dataset -o ./my_results
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input data folder containing your dataset"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Path to output directory (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "-c", "--config", 
        default="configs/default.yaml", 
        help="Configuration file path (default: configs/default.yaml)"
    )
    
    parser.add_argument(
        "--single",
        choices=["traditional", "custom_nn", "pretrained"],
        default=None,
        help="Run only a single iteration approach instead of all 3"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input directory not found: {input_path}")
        print("Make sure your dataset directory exists and contains:")
        print("  - description.txt (problem description)")
        print("  - *.csv files (your data)")
        sys.exit(1)
    
    # Check for required files
    description_file = input_path / "description.txt"
    if not description_file.exists():
        print(f"❌ Error: Required file 'description.txt' not found in {input_path}")
        print("Please create a description.txt file describing your ML problem.")
        sys.exit(1)
    
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"❌ Error: No CSV files found in {input_path}")
        print("Please ensure your dataset contains at least one CSV file.")
        sys.exit(1)
    
    print("🤖 iML Multi-Iteration AutoML")
    print("=" * 50)
    print(f"📂 Input Dataset: {input_path.resolve()}")
    print(f"📋 Description File: {'✓' if description_file.exists() else '✗'}")
    print(f"📊 CSV Files Found: {len(csv_files)}")
    print(f"⚙️  Configuration: {args.config}")
    
    if args.single:
        print(f"🎯 Mode: Single Iteration ({args.single})")
        checkpoint_mode = "full"
        single_iteration = args.single
    else:
        print("🎯 Mode: Multi-Iteration (all 3 approaches)")
        print("   • Traditional ML (XGBoost, LightGBM, CatBoost)")
        print("   • Custom Neural Networks (PyTorch)")
        print("   • Pretrained Models (HuggingFace)")
        checkpoint_mode = "multi-iteration"
        single_iteration = None
    
    print("\n🚀 Starting AutoML Pipeline...")
    print("=" * 50)
    
    try:
        # Run the pipeline
        run_automl_pipeline(
            input_data_folder=str(input_path),
            output_folder=args.output,
            config_path=args.config,
            checkpoint_mode=checkpoint_mode,
            single_iteration=single_iteration,
        )
        
        print("\n" + "=" * 50)
        print("✅ AutoML Pipeline Completed Successfully!")
        
        if not args.single:
            print("\n📊 Results Summary:")
            print("• Check llm_comparison_results.json for detailed analysis")
            print("• Best solution copied to final_submission/ folder")
            print("• All 3 iterations available in separate folders")
        
        print(f"\n📁 Output saved in: runs/ directory")
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is set correctly")
        print("2. Verify your dataset format and description.txt")
        print("3. Check the logs in the output directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
