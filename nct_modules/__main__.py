#!/usr/bin/env python3
"""
NCT Dashboard CLI Entry Point
Launch the NeuroConscious Transformer visualization dashboard.

Usage:
    nct-dashboard
    python -m nct_modules
"""

import sys
import os
import subprocess


def main():
    """Main entry point for nct-dashboard command."""
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("\nTo install dashboard dependencies, run:")
        print("  pip install neuroconscious-transformer[dashboard]")
        sys.exit(1)

    # Find the dashboard file in the package
    dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'visualization', 'nct_dashboard.py')
    
    # Also check current directory for development mode
    if not os.path.exists(dashboard_path):
        dashboard_path = os.path.join(os.path.dirname(__file__), 'visualization', 'nct_dashboard.py')
    
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        print("\nMake sure the visualization/nct_dashboard.py file exists.")
        sys.exit(1)

    # Run streamlit with the dashboard
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path] + sys.argv[1:])


if __name__ == "__main__":
    main()
