import subprocess
import sys
import os

def run_streamlit_debug():
    # Add your app's directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import your streamlit app (don't include .py extension)
    import app  # Corrected: remove .py extension
    
    # Run streamlit (use the actual filename with .py)
    subprocess.run([
        sys.executable, 
        "-m", "streamlit", 
        "run", 
        "app.py",  # Corrected: use your actual filename
        "--server.port=8501"
    ])

if __name__ == "__main__":
    run_streamlit_debug()