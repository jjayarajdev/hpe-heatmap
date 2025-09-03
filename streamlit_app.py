"""
HPE Talent Intelligence Platform - Main Entry Point for Streamlit Cloud
This file serves as the entry point for Streamlit Cloud deployment
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the main app
from app.complete_enhanced_app import main

if __name__ == "__main__":
    main()