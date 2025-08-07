# src/preprocess.py

from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_and_save_data():
    # Load California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Save to CSV
    df.to_csv("data/housing.csv", index=False)
    print("✅ California Housing dataset saved to data/housing.csv")

if __name__ == "__main__":
    load_and_save_data()

