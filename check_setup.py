#!/usr/bin/env python3
"""
Setup validation script for iML AutoML Framework
Checks if the environment is properly configured for running multi-iteration AutoML.
"""
import os
import sys
from pathlib import Path
import importlib

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        return False, f"Python {version.major}.{version.minor} (requires Python 3.8+)"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'torch',
        'transformers',
        'xgboost',
        'lightgbm',
        'catboost',
        'langchain',
        'omegaconf',
        'ydata_profiling',
        'rich'
    ]
    
    results = {}
    for package in required_packages:
        try:
            module = importlib.import_module(package.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            results[package] = (True, version)
        except ImportError:
            results[package] = (False, "Not installed")
    
    return results

def check_api_keys():
    """Check if LLM API keys are configured."""
    api_keys = {
        'GEMINI_API_KEY': 'Google Gemini (default)',
        'OPENAI_API_KEY': 'OpenAI GPT',
        'ANTHROPIC_API_KEY': 'Anthropic Claude',
        'AWS_DEFAULT_REGION': 'AWS Bedrock'
    }
    
    results = {}
    for key, provider in api_keys.items():
        value = os.getenv(key)
        if value:
            # Mask the key for security
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            results[provider] = (True, masked)
        else:
            results[provider] = (False, "Not set")
    
    return results

def check_file_structure():
    """Check if the project file structure is correct."""
    required_paths = [
        'src/iML',
        'src/iML/agents',
        'src/iML/core', 
        'src/iML/prompts',
        'src/iML/llm',
        'configs/default.yaml',
        'requirements.txt'
    ]
    
    results = {}
    for path in required_paths:
        full_path = Path(path)
        results[path] = full_path.exists()
    
    return results

def main():
    """Run all setup checks."""
    print("🤖 iML AutoML Framework - Setup Validation")
    print("=" * 60)
    
    # Check Python version
    print("\n📋 Python Version:")
    py_ok, py_info = check_python_version()
    status = "✅" if py_ok else "❌"
    print(f"  {status} {py_info}")
    
    # Check dependencies
    print("\n📦 Dependencies:")
    deps = check_dependencies()
    all_deps_ok = True
    for package, (installed, version) in deps.items():
        status = "✅" if installed else "❌"
        print(f"  {status} {package:<20} {version}")
        if not installed:
            all_deps_ok = False
    
    # Check API keys
    print("\n🔑 API Keys:")
    api_keys = check_api_keys()
    any_key_set = False
    for provider, (configured, value) in api_keys.items():
        status = "✅" if configured else "⚠️ "
        print(f"  {status} {provider:<25} {value}")
        if configured:
            any_key_set = True
    
    # Check file structure
    print("\n📁 File Structure:")
    files = check_file_structure()
    all_files_ok = True
    for path, exists in files.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {path}")
        if not exists:
            all_files_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Setup Summary:")
    
    checks = [
        ("Python Version", py_ok),
        ("Dependencies", all_deps_ok),
        ("API Keys", any_key_set),
        ("File Structure", all_files_ok)
    ]
    
    all_good = True
    for check_name, check_ok in checks:
        status = "✅" if check_ok else "❌"
        print(f"  {status} {check_name}")
        if not check_ok:
            all_good = False
    
    if all_good:
        print("\n🎉 Setup Complete! You're ready to run iML AutoML.")
        print("\nQuick start:")
        print("  python run_multi_iteration.py -i ./your_dataset")
    else:
        print("\n⚠️  Setup Issues Found:")
        if not py_ok:
            print("  • Upgrade to Python 3.8 or higher")
        if not all_deps_ok:
            print("  • Install missing dependencies: pip install -r requirements.txt")
        if not any_key_set:
            print("  • Set up at least one API key (GEMINI_API_KEY recommended)")
        if not all_files_ok:
            print("  • Make sure you're running from the project root directory")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
