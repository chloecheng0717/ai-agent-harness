"""
Test Installation Script

This script verifies that the project is properly installed and configured.
"""

import sys
import subprocess
import importlib


def check_python_version():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 12):
        print("ERROR: Python 3.12 or higher is required")
        return False
    return True


def check_dependencies():
    """Check if dependencies are installed."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
        "redis",
        "structlog",
        "langgraph",
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r backend/requirements/base.txt")
        return False
    
    return True


def check_application_imports():
    """Check if application modules can be imported."""
    try:
        from app.config import get_settings
        from app.models.task import TaskCreate
        from app.agents.base import BaseAgent
        from app.tools.base import BaseTool
        from app.memory.base import BaseMemory
        print("✓ Application modules import successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import application modules: {e}")
        return False


def main():
    """Run all checks."""
    print("Testing AI Agent Harness installation...\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Application Imports", check_application_imports),
    ]
    
    results = []
    for name, check in checks:
        print(f"\n{name}:")
        result = check()
        results.append(result)
    
    print("\n" + "="*50)
    if all(results):
        print("SUCCESS: All checks passed!")
        print("\nYou can now run the application with:")
        print("  cd backend")
        print("  python run.py")
        return 0
    else:
        print("FAILED: Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())