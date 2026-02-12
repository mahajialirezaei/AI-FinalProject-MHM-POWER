"""
Simple script to run the Streamlit UI
Run this from the project root: python run_ui.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the UI file path
    project_root = Path(__file__).parent
    ui_file = project_root / "src" / "ui" / "streamlit.py"
    
    if not ui_file.exists():
        print(f"Error: UI file not found at {ui_file}")
        sys.exit(1)
    
    # Run streamlit
    print(f"Starting Streamlit UI: {ui_file}")
    print("The app will open in your default browser.")
    print("Press Ctrl+C to stop the server.")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(ui_file)
    ])
