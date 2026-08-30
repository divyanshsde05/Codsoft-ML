# =============================================================================
# TASK 3: CUSTOMER CHURN PREDICTION (Tabular Pipeline)
# Uses a dummy dataset that mirrors a real Telco Churn CSV.
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  To use YOUR real data, replace the dummy_df block below with:           │
# │      df = pd.read_csv(r"path\to\WA_Fn-UseC_-Telco-Customer-Churn.csv") │
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
)
from sklearn.impute import SimpleImputer


# ── REUSABLE TABULAR PIPELINE ─────────────────────────────────────────────────
def run_tabular_pipeline(
    df, feature_cols, label_col,
    test_size=0.20, random_state=42,
    handle_imbalance=False,
    n_estimators=150,
    task_name="Tabular Task",
):
    """
    Universal tabular classification pipeline using Random Forest.

    handle_imbalance=True → class_weight='balanced' (use for fraud detection).
    """
    print(f"\n{'='*60}\n  {task_name}\n{'='*60}")

    df = df.copy()

    # ── Preprocessing ─────────────────────────────────────────────────────────
    num_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    le_map = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_map[c] = le

    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df[feature_cols].values
    y = df[label_col].values

    print(f"  Samples    : {len(df):,}")
    print(f"  Features   : {len(feature_cols)}")
    print(f"  Class dist :\n{pd.Series(y).value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cw = "balanced" if handle_imbalance else None
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=cw,
        random_state=random_state,
        n_jobs=-1,
        max_depth=15,
        min_samples_leaf=2,
    )
    print("\n  Training Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    classes = sorted(np.unique(y))

    # ROC-AUC
    if len(classes) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"  ROC-AUC : {auc:.4f}")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0,1],[0,1],"--", color="gray")
        ax.set(xlabel="FPR", ylabel="TPR", title=f"{task_name}\nROC Curve")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "roc_curve.png"), dpi=120)
        plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax2, colorbar=False)
    ax2.set_title(f"{task_name}\nConfusion Matrix (Acc: {acc*100:.2f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"), dpi=120)
    plt.show()

    # Feature Importance
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_cols, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(20)
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    sns.barplot(data=fi, x="importance", y="feature", palette="mako", ax=ax3)
    ax3.set_title(f"{task_name}\nTop-20 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "feature_importance.png"), dpi=120)
    plt.show()

    # Churn rate EDA
    orig_df = df.copy()
    orig_df["__pred__"] = y_pred
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
    pd.Series(y).value_counts().plot(kind="pie", ax=axes[0],
                                     autopct="%1.1f%%", colors=["#3498db","#e74c3c"],
                                     labels=["No Churn","Churn"])
    axes[0].set_title("Actual Class Distribution")
    pd.Series(y_pred).value_counts().plot(kind="pie", ax=axes[1],
                                          autopct="%1.1f%%", colors=["#3498db","#e74c3c"],
                                          labels=["No Churn","Churn"])
    axes[1].set_title("Predicted Class Distribution")
    plt.suptitle(f"{task_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "class_distribution.png"), dpi=120)
    plt.show()

    return model, {"accuracy": acc}


# ── DUMMY DATASET ──────────────────────────────────────────────────────────────
# !! REPLACE WITH REAL DATA !!
# df = pd.read_csv(r"D:\codsoft\WA_Fn-UseC_-Telco-Customer-Churn.csv")

def make_dummy_churn_data(n=5_000, churn_rate=0.27, seed=42):
    rng = np.random.default_rng(seed)
    n_churn = int(n * churn_rate)
    n_stay  = n - n_churn

    def make_block(size, churn):
        return pd.DataFrame({
            "tenure":          rng.integers(1, 72, size),
            "MonthlyCharges":  rng.uniform(20, 120, size),
            "TotalCharges":    rng.uniform(20, 8000, size),
            "SeniorCitizen":   rng.integers(0, 2, size),
            "gender":          rng.choice(["Male","Female"], size),
            "Partner":         rng.choice(["Yes","No"], size),
            "Dependents":      rng.choice(["Yes","No"], size),
            "PhoneService":    rng.choice(["Yes","No"], size),
            "MultipleLines":   rng.choice(["Yes","No","No phone service"], size),
            "InternetService": rng.choice(["DSL","Fiber optic","No"], size),
            "OnlineSecurity":  rng.choice(["Yes","No","No internet service"], size),
            "TechSupport":     rng.choice(["Yes","No","No internet service"], size),
            "Contract":        rng.choice(["Month-to-month","One year","Two year"], size),
            "PaperlessBilling":rng.choice(["Yes","No"], size),
            "PaymentMethod":   rng.choice(["Electronic check","Mailed check",
                                           "Bank transfer","Credit card"], size),
            "Churn":           [1 if churn else 0]*size,
        })

    df = pd.concat([make_block(n_stay, 0), make_block(n_churn, 1)],
                   ignore_index=True).sample(frac=1, random_state=seed)
    return df


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building dummy Customer Churn dataset (replace with real CSV)...")
    df = make_dummy_churn_data(n=5_000)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate   : {df['Churn'].mean()*100:.1f}%")
    print(df.head(3))

    feature_cols = [c for c in df.columns if c != "Churn"]

    model, results = run_tabular_pipeline(
        df=df,
        feature_cols=feature_cols,
        label_col="Churn",
        handle_imbalance=True,   # set False if dataset is already balanced
        n_estimators=150,
        task_name="Task 3: Customer Churn Prediction",
    )

    print(f"\n✅ Task 3 Complete!  Accuracy: {results['accuracy']*100:.2f}%")
