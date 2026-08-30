# Spam SMS Detection

This project detects spam messages using natural language processing and classical machine learning.

## Goal

Classify SMS messages as `ham` or `spam`.

## Model

- Text preprocessing
- TF-IDF feature extraction
- SMOTE to handle class imbalance
- LinearSVC classifier

## Files

- `spam_sms_classification.ipynb` — main notebook
- `../spam.csv` — SMS dataset

## Run

1. Open the notebook in Jupyter or VS Code.
2. Ensure `spam.csv` exists in the repository root.
3. Run all cells in order.

## Expected Output

- Class imbalance visualization
- Balanced training data after SMOTE
- Accuracy and classification report
- Confusion matrix
- Manual sample testing for spam/ham examples
