#!/usr/bin/env python3
"""
Simple test script to verify multi-iteration pipeline implementation.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all modified modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from iML.core.manager import Manager
        print("✓ Manager imported successfully")
        
        from iML.agents.guideline_agent import GuidelineAgent
        print("✓ GuidelineAgent imported successfully")
        
        from iML.agents.preprocessing_coder_agent import PreprocessingCoderAgent
        print("✓ PreprocessingCoderAgent imported successfully")
        
        from iML.agents.modeling_coder_agent import ModelingCoderAgent
        print("✓ ModelingCoderAgent imported successfully")
        
        from iML.agents.assembler_agent import AssemblerAgent
        print("✓ AssemblerAgent imported successfully")
        
        from iML.prompts.guideline_prompt import GuidelinePrompt
        print("✓ GuidelinePrompt imported successfully")
        
        from iML.prompts.preprocessing_coder_prompt import PreprocessingCoderPrompt
        print("✓ PreprocessingCoderPrompt imported successfully")
        
        from iML.prompts.modeling_coder_prompt import ModelingCoderPrompt
        print("✓ ModelingCoderPrompt imported successfully")
        
        from iML.prompts.assembler_prompt import AssemblerPrompt
        print("✓ AssemblerPrompt imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_manager_methods():
    """Test that Manager has required methods."""
    print("\nTesting Manager methods...")
    
    try:
        from iML.core.manager import Manager
        
        # Check if multi-iteration methods exist
        if hasattr(Manager, 'run_pipeline_multi_iteration'):
            print("✓ run_pipeline_multi_iteration method exists")
        else:
            print("✗ run_pipeline_multi_iteration method missing")
            return False
            
        if hasattr(Manager, '_run_shared_analysis'):
            print("✓ _run_shared_analysis method exists")
        else:
            print("✗ _run_shared_analysis method missing")
            return False
            
        if hasattr(Manager, '_run_iteration_pipeline'):
            print("✓ _run_iteration_pipeline method exists")
        else:
            print("✗ _run_iteration_pipeline method missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Manager method test failed: {e}")
        return False

def test_prompt_methods():
    """Test that prompt classes have iteration guidance methods."""
    print("\nTesting prompt iteration methods...")
    
    try:
        from iML.prompts.guideline_prompt import GuidelinePrompt
        from iML.prompts.preprocessing_coder_prompt import PreprocessingCoderPrompt
        from iML.prompts.modeling_coder_prompt import ModelingCoderPrompt
        from iML.prompts.assembler_prompt import AssemblerPrompt
        
        # Create dummy instances to test methods
        class DummyManager:
            def save_and_log_states(self, *args, **kwargs):
                pass
        
        manager = DummyManager()
        
        # Test GuidelinePrompt
        gp = GuidelinePrompt(llm_config={}, manager=manager)
        if hasattr(gp, '_get_algorithm_constraint'):
            print("✓ GuidelinePrompt._get_algorithm_constraint exists")
            # Test method
            constraint = gp._get_algorithm_constraint("traditional")
            if "TRADITIONAL ML ALGORITHMS" in constraint:
                print("✓ Traditional algorithm constraint working")
        else:
            print("✗ GuidelinePrompt._get_algorithm_constraint missing")
            return False
        
        # Test PreprocessingCoderPrompt
        pcp = PreprocessingCoderPrompt(llm_config={}, manager=manager)
        if hasattr(pcp, '_get_iteration_guidance'):
            print("✓ PreprocessingCoderPrompt._get_iteration_guidance exists")
        else:
            print("✗ PreprocessingCoderPrompt._get_iteration_guidance missing")
            return False
        
        # Test ModelingCoderPrompt
        mcp = ModelingCoderPrompt(llm_config={}, manager=manager)
        if hasattr(mcp, '_get_iteration_guidance'):
            print("✓ ModelingCoderPrompt._get_iteration_guidance exists")
        else:
            print("✗ ModelingCoderPrompt._get_iteration_guidance missing")
            return False
        
        # Test AssemblerPrompt
        ap = AssemblerPrompt(llm_config={}, manager=manager)
        if hasattr(ap, '_get_iteration_guidance'):
            print("✓ AssemblerPrompt._get_iteration_guidance exists")
        else:
            print("✗ AssemblerPrompt._get_iteration_guidance missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt method test failed: {e}")
        return False

def test_cli_interface():
    """Test that CLI supports multi-iteration mode."""
    print("\nTesting CLI interface...")
    
    try:
        # Read the run.py file to check for multi-iteration support
        with open("run.py", "r") as f:
            content = f.read()
        
        if "multi-iteration" in content:
            print("✓ CLI supports multi-iteration mode")
        else:
            print("✗ CLI missing multi-iteration support")
            return False
        
        # Check main_runner.py
        with open("src/iML/main_runner.py", "r") as f:
            content = f.read()
        
        if "run_pipeline_multi_iteration" in content:
            print("✓ main_runner calls multi-iteration pipeline")
        else:
            print("✗ main_runner missing multi-iteration call")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Multi-Iteration Pipeline Implementation Test ===\n")
    
    tests = [
        test_imports,
        test_manager_methods,
        test_prompt_methods,
        test_cli_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Multi-iteration implementation looks good.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


