import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import time

# --- Configuration ---
# Set paths and names to avoid hardcoding them multiple times
DATA_PATH = "data/housing.csv"
EXPERIMENT_NAME = "california-housing"
REGISTERED_MODEL_NAME = "CaliforniaHousingModel"


def main():
    """
    Main function to run the training pipeline.
    - Loads data
    - Trains multiple models
    - Finds the best model based on MSE
    - Registers the best model and promotes it to Production
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # --- 1. Load and Prepare Data ---
    print("Loading and preparing data...")
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=["MedHouseVal"])
    y = data["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    feature_names = list(X.columns)
    print("✅ Data ready.")

    # --- 2. Train and Evaluate Models ---
    # Define models in a dictionary to easily iterate through them
    models_to_train = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    best_mse = float("inf")
    best_run_id = None

    print("\nStarting model training loop...")
    for model_name, model in models_to_train.items():
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            print(f"--- Training {model_name} (Run ID: {run_id}) ---")

            # Fit model and make predictions
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            # Log parameters, metrics, and artifacts
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("mse", f"{mse:.6f}")
            mlflow.log_dict({"features": feature_names}, "features.json")

            # Infer signature and log the model
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(1),
            )

            print(f"✅ {model_name} MSE: {mse:.4f}")

            # Check if this is the best model so far
            if mse < best_mse:
                best_mse = mse
                best_run_id = run_id
                print(f"⭐ New best model found: {model_name}")

    # FIX: Re-formatted long f-string
    best_model_message = (
        f"\n🏆 Best model is from Run ID: {best_run_id} "
        f"with MSE: {best_mse:.4f}"
    )
    print(best_model_message)

    # --- 3. Register and Promote the Best Model ---
    if not best_run_id:
        raise RuntimeError("Training failed, no best model was found.")

    # FIX: Re-formatted long f-string
    registration_message = (
        f"\nRegistering best model under the name: "
        f"'{REGISTERED_MODEL_NAME}'..."
    )
    print(registration_message)
    model_uri = f"runs:/{best_run_id}/model"
    try:
        registered_model_version = mlflow.register_model(
            model_uri=model_uri, name=REGISTERED_MODEL_NAME
        )
        print(
            f"✅ Model registered successfully. "
            f"Version: {registered_model_version.version}"
        )
    except Exception as e:
        print(f"🚨 Model registration failed: {e}")
        return

    # --- 4. Wait for Model to be 'READY' and Promote ---
    print("Waiting for model version to become 'READY'...")
    for _ in range(10):  # Wait up to 10 seconds
        model_version_details = client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=registered_model_version.version
        )
        if model_version_details.status == "READY":
            print("✅ Model is READY.")
            break
        time.sleep(1)

    if model_version_details.status != "READY":
        error_message = (
            "🚨 Model version did not become ready in time. "
            "Aborting promotion."
        )
        print(error_message)
        return

    promotion_message = (
        f"Promoting model version {registered_model_version.version} "
        "to 'Production' stage..."
    )
    print(promotion_message)
    try:
        # Optional: move old Production models to Archive
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=registered_model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print("✅ Model successfully promoted to Production!")
    except Exception as e:
        print(f"🚨 Model promotion failed: {e}")


if __name__ == "__main__":
    main()
