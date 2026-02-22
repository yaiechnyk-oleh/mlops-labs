"""
Download the Twitter Sentiment Analysis dataset.

Option 1: Manual download from Kaggle
    1. Go to: https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech
    2. Download the dataset
    3. Extract and place CSV file as: data/raw/twitter.csv

Option 2: Use Kaggle API (requires ~/.kaggle/kaggle.json)
    pip install kaggle
    kaggle datasets download -d arkhoshghalb/twitter-sentiment-analysis-hatred-speech -p data/raw/ --unzip

Option 3: Run this script (uses opendatasets)
    pip install opendatasets
    python src/download_data.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def check_data_exists():
    """Check if the dataset already exists."""
    csv_path = os.path.join(DATA_DIR, "twitter.csv")
    if os.path.exists(csv_path):
        print(f"✅ Dataset already exists at: {csv_path}")
        return True
    # Check for other CSV files that might be the dataset
    for f in os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []:
        if f.endswith(".csv"):
            print(f"Found CSV file: {os.path.join(DATA_DIR, f)}")
            print(f"Rename it to 'twitter.csv' if it's the correct dataset.")
            return True
    return False


def download_with_opendatasets():
    """Download using the opendatasets library."""
    try:
        import opendatasets as od
    except ImportError:
        print("Installing opendatasets...")
        os.system(f"{sys.executable} -m pip install opendatasets")
        import opendatasets as od

    url = "https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech"
    od.download(url, data_dir=DATA_DIR)
    print(f"\n✅ Dataset downloaded to: {DATA_DIR}")
    print("You may need to rename the file to 'twitter.csv'.")


if __name__ == "__main__":
    if check_data_exists():
        sys.exit(0)
    
    print("Dataset not found. Attempting download...")
    print("You will need your Kaggle credentials (username + API key).")
    print("Get your key from: https://www.kaggle.com/settings → API → Create New Token\n")
    
    download_with_opendatasets()
