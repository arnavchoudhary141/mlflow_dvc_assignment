import mlflow
import mlflow.sklearn
import subprocess
from sklearn.ensemble import RandomForestClassifier


# 1. Get the current data version from DVC
with open("data/train.csv.dvc", "r") as f:
    for line in f:
        if "md5:" in line:
            dvc_version = line.split("md5:")[1].strip()
            break

# 2. Start MLflow experiment
mlflow.set_experiment("My_First_ML_Project")

with mlflow.start_run():
    # Log the data version as a parameter
    mlflow.log_param("dataset_version", dvc_version)

    # Define and log model parameters
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)

    # Dummy training, as specified in the lab
    model = RandomForestClassifier(n_estimators=n_estimators)

    # model.fit(X_train, y_train)

    # Log a dummy metric
       # Log a dummy metric
    mlflow.log_metric("accuracy", 0.95)

    # Log the model itself
    mlflow.sklearn.log_model(model, name="random_forest_model")

    # Create and log an artifact file
    with open("mlflow_artifact.txt", "w") as f:
        f.write("MLflow + DVC pipeline executed successfully.\n")
        f.write(f"Dataset version: {dvc_version}\n")
        f.write(f"Accuracy: 0.95\n")
        f.write(f"Number of estimators: {n_estimators}\n")

    mlflow.log_artifact("mlflow_artifact.txt")

print("Experiment tracked in MLflow!")