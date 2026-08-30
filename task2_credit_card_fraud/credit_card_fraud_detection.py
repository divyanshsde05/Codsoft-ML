# =============================================================================
# TASK 2: CREDIT CARD FRAUD DETECTION (Tabular Pipeline)
# Uses a dummy dataset that mirrors real Kaggle creditcard.csv structure.
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  To use YOUR real data, replace the dummy_df block below with:           │
# │      df = pd.read_csv(r"path\to\creditcard.csv")                         │
# │  The CSV should have columns V1..V28, Amount, Time, Class               │
# └──────────────────────────────────────────────────────────────────────────┘
# =============================================================================

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.impute import SimpleImputer


# ── REUSABLE TABULAR PIPELINE ─────────────────────────────────────────────────
def run_tabular_pipeline(
    df, feature_cols, label_col,
    test_size=0.20, random_state=42,
    handle_imbalance=False,          # Set True for fraud detection
    n_estimators=200,
    task_name="Tabular Task",
):
    """
    Universal tabular classification pipeline using Random Forest.

    Parameters
    ----------
    df               : Input DataFrame.
    feature_cols     : List of column names used as features.
    label_col        : Target column name.
    handle_imbalance : If True, uses class_weight='balanced' (ideal for fraud).
    n_estimators     : Number of trees in the Random Forest.
    task_name        : Display label for charts/logs.

    Returns
    -------
    model   : Fitted RandomForestClassifier.
    results : Dict with accuracy + report string.
    """
    print(f"\n{'='*60}\n  {task_name}\n{'='*60}")

    df = df.copy()

    # ── Preprocessing ─────────────────────────────────────────────────────────
    # 1) Separate numeric and categorical columns
    num_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # 2) Encode categorical columns
    le_map = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_map[c] = le

    # 3) Impute missing values
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # 4) Scale numeric columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df[feature_cols].values
    y = df[label_col].values

    print(f"  Samples    : {len(df):,}")
    print(f"  Features   : {len(feature_cols)}")
    print(f"  Class dist :\n{pd.Series(y).value_counts().to_string()}")
    print(f"  Imbalance  : {'BALANCED (class_weight)' if handle_imbalance else 'default'}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    cw = "balanced" if handle_imbalance else None
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=cw,
        random_state=random_state,
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=2,
    )
    print("\n  Training Random Forest...")
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    classes = sorted(np.unique(y))

    # ROC-AUC
    if len(classes) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ap  = average_precision_score(y_test, y_prob)
        print(f"  ROC-AUC  : {auc:.4f} | Avg Precision : {ap:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0,1],[0,1],"--", color="gray")
        ax.set(xlabel="FPR", ylabel="TPR", title=f"{task_name}\nROC Curve")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "roc_curve.png"), dpi=120)
        plt.show()

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(rec, prec, color="steelblue", lw=2, label=f"AP = {ap:.3f}")
        ax2.set(xlabel="Recall", ylabel="Precision", title=f"{task_name}\nPrecision-Recall Curve")
        ax2.legend(); plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "pr_curve.png"), dpi=120)
        plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax3, colorbar=False)
    ax3.set_title(f"{task_name}\nConfusion Matrix (Acc: {acc*100:.2f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"), dpi=120)
    plt.show()

    # Feature Importance
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_cols, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(20)
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    sns.barplot(data=fi, x="importance", y="feature", palette="viridis", ax=ax4)
    ax4.set_title(f"{task_name}\nTop-20 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "feature_importance.png"), dpi=120)
    plt.show()

    return model, {"accuracy": acc}


# ── DATASET ────────────────────────────────────────────────────────────────────
# !! REPLACE THIS BLOCK WITH YOUR REAL DATA !!
# df = pd.read_csv(r"D:\codsoft\creditcard.csv")

def make_dummy_fraud_data(n=10_000, fraud_rate=0.01, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud  = int(n * fraud_rate)
    n_normal = n - n_fraud

    normal = pd.DataFrame(rng.standard_normal((n_normal, 28)),
                          columns=[f"V{i}" for i in range(1, 29)])
    fraud  = pd.DataFrame(rng.standard_normal((n_fraud, 28)) + 3,
                          columns=[f"V{i}" for i in range(1, 29)])

    normal["Amount"] = rng.exponential(50, n_normal)
    fraud["Amount"]  = rng.exponential(150, n_fraud)
    normal["Time"]   = rng.uniform(0, 172800, n_normal)
    fraud["Time"]    = rng.uniform(0, 172800, n_fraud)
    normal["Class"]  = 0
    fraud["Class"]   = 1

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=seed)
    return df


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building dummy Credit Card Fraud dataset (replace with real CSV)...")
    df = make_dummy_fraud_data(n=10_000, fraud_rate=0.02)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate   : {df['Class'].mean()*100:.2f}%")

    feature_cols = [c for c in df.columns if c != "Class"]

    model, results = run_tabular_pipeline(
        df=df,
        feature_cols=feature_cols,
        label_col="Class",
        handle_imbalance=True,   # CRITICAL for fraud — keeps recall high
        n_estimators=200,
        task_name="Task 2: Credit Card Fraud Detection",
    )

    print(f"\n✅ Task 2 Complete!  Accuracy: {results['accuracy']*100:.2f}%")
