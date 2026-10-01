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
    mlflow.log_metric("accuracy", 0.95)

    # Log the model itself
    mlflow.sklearn.log_model(model, "random_forest_model")

    print("Experiment tracked in MLflow!")
