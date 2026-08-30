# Customer Churn Prediction

This project predicts whether a customer is likely to churn using a tabular machine learning workflow.

## Goal

Classify customers as either likely to stay or likely to leave based on customer attributes such as tenure, charges, contract type, internet service, and payment method.

## Model

- Data preprocessing
- Label encoding for categorical variables
- Median imputation and feature scaling
- Random Forest classifier
- Evaluation using accuracy, confusion matrix, and ROC-AUC

## Files

- `customer_churn_prediction.ipynb` — interactive notebook version
- `customer_churn_prediction.py` — script version

## Run

1. Activate the virtual environment.
2. Install dependencies from the root `requirements.txt` file.
3. Run the notebook or execute:

```bash
python task3_customer_churn/customer_churn_prediction.py
```

## Notes

- The notebook includes a dummy synthetic dataset that mirrors the Telco churn structure.
- To use the real Kaggle CSV, replace the generated dataset block with:

```python
df = pd.read_csv(r"path\to\WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

## Expected Output

- Churn distribution plot
- Model accuracy and classification report
- Confusion matrix
- ROC curve
- Feature importance chart
