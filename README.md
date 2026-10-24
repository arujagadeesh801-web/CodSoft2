# Credit Card Fraud Detection

A Python machine-learning pipeline for training and evaluating classifiers that identify potentially fraudulent credit card transactions.

## Requirements

- Python 3
- pip
- The training and test CSV files in the `data/` directory

## Setup

Run these commands from the project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, you can install dependencies without activating it:

```powershell
py -m pip install -r requirements.txt
```

## Run

Train the default Random Forest classifier and evaluate it on the test data:

```powershell
py main.py
```

Choose a classifier with `--model_type`:

```powershell
py main.py --model_type logistic
py main.py --model_type decision_tree
py main.py --model_type random_forest
```

To use CSV files at different locations, provide both paths:

```powershell
py main.py --train_path "data/fraudTrain.csv" --test_path "data/fraudTest.csv"
```

The supported classifier values are `logistic`, `decision_tree`, and `random_forest`. The default is `random_forest`.

## What the pipeline does

1. Loads the training and test CSV files.
2. Removes selected identifier and text columns and one-hot encodes categorical features.
3. Scales the features and trains the selected classifier.
4. Saves the trained model to `models/model.pkl` and the scaler to `models/scaler.pkl`.
5. Prints a classification report, confusion matrix, ROC-AUC score (when available), and accuracy.

## Project structure

```text
main.py                  Command-line entry point
requirements.txt         Python dependencies
data/                    Training and test CSV files
models/                  Saved model and scaler
src/preprocessing.py     Data loading and preprocessing
src/train.py             Model training and artifact saving
src/evaluate.py          Model evaluation
```
