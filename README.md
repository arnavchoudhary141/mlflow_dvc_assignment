# MLflow + DVC Assignment

This project follows the uploaded MLflow + DVC lab document.

## Objective

- DVC versions the dataset.
- Git tracks the DVC metadata.
- MLflow tracks the dataset version, model parameter, metric, and model.
- MLflow UI is used to view the experiment.

## Project structure

```text
mlflow_dvc_assignment/
├── data/
│   └── train.csv
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run on macOS

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Initialize Git and DVC

```bash
git init
dvc init
```

### 4. Add the dataset to DVC

```bash
dvc add data/train.csv
```

This creates:

```text
data/train.csv.dvc
```

and updates `.gitignore`.

### 5. Commit the DVC metadata to Git

```bash
git add data/train.csv.dvc .gitignore
git commit -m "Add v1 of training data"
```

### 6. Run the MLflow experiment

```bash
python train.py
```

The script logs:

- `dataset_version`
- `n_estimators = 100`
- `accuracy = 0.95`
- `random_forest_model`

### 7. Start the MLflow UI

Open another terminal in the same project directory, activate the environment, then run:

```bash
source .venv/bin/activate
mlflow ui
```

Open:

```text
http://localhost:5000
```

## Important

The Python code intentionally follows the lab document's example, including the dummy metric and commented-out training line.

If the installed DVC version does not provide `dvc hash`, run:

```bash
dvc add data/train.csv
```

first and use the dataset hash recorded in `data/train.csv.dvc` as the dataset version value in MLflow.

## GitHub submission

After verifying the project locally:

```bash
git status
git add .
git commit -m "Complete MLflow and DVC assignment"
```

Then create a GitHub repository and connect it:

```bash
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git branch -M main
git push -u origin main
```
