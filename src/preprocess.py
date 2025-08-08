from sklearn.datasets import fetch_california_housing
import os


def load_and_save_data():
    """
    Load California Housing dataset and save it to data/housing.csv
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/housing.csv", index=False)
    print("✅ California Housing dataset saved to data/housing.csv")


if __name__ == "__main__":
    load_and_save_data()
