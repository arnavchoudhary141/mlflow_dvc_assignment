"""MLflow and DVC training pipeline."""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier


# Get the current data version from DVC.
with open("data/train.csv.dvc", "r", encoding="utf-8") as file:
    for line in file:
        if "md5:" in line:
            dvc_version = line.split("md5:")[1].strip()
            break

# Start MLflow experiment.
mlflow.set_experiment("My_First_ML_Project")

with mlflow.start_run():
    # Log the data version as a parameter.
    mlflow.log_param("dataset_version", dvc_version)

    # Define and log model parameters.
    N_ESTIMATORS = 100
    mlflow.log_param("n_estimators", N_ESTIMATORS)

    # Create the Random Forest model.
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS)

    # Log a dummy metric as specified in the lab.
    mlflow.log_metric("accuracy", 0.95)

    # Log the model.
    mlflow.sklearn.log_model(model, name="random_forest_model")

    # Create and log an artifact file.
    with open("mlflow_artifact.txt", "w", encoding="utf-8") as file:
        file.write("MLflow + DVC pipeline executed successfully.\n")
        file.write(f"Dataset version: {dvc_version}\n")
        file.write("Accuracy: 0.95\n")
        file.write(f"Number of estimators: {N_ESTIMATORS}\n")

    mlflow.log_artifact("mlflow_artifact.txt")

print("Experiment tracked in MLflow!")
