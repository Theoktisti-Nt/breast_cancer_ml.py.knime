#pip install scikit-learn pandas numpy matplotlib joblib

# --- ΜΗ ΔΙΑΔΡΑΣΤΙΚΟ BACKEND ΓΙΑ PLOTS (χωρίς tkinter) ---
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

# --- IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, learning_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    roc_auc_score, average_precision_score, precision_recall_curve, f1_score
)
from sklearn.inspection import permutation_importance

# =============== ΡΥΘΜΙΣΕΙΣ ΔΙΑΔΡΟΜΩΝ ===============
BASE_DIR = r"C:\cancer.ml.py.knime"
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs_full")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =============== LOAD DATA ===============
df = pd.read_csv(DATA_PATH)
assert "diagnosis" in df.columns, "Λείπει η στήλη 'diagnosis' (B/M)."
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Labels: B->0, M->1
le = LabelEncoder()
y = le.fit_transform(df["diagnosis"])  # B=0, M=1
X = df.drop(columns=["diagnosis"])
feature_names = X.columns.tolist()

# Class balance
counts = np.bincount(y)
print(f"Class balance: Benign={counts[0]}, Malignant={counts[1]} ({counts[1]/sum(counts):.2%} malignant)")
with open(os.path.join(OUT_DIR, "class_balance.txt"), "w", encoding="utf-8") as f:
    f.write(f"Benign={counts[0]}, Malignant={counts[1]} ({counts[1]/sum(counts):.2%} malignant)\n")

# =============== TRAIN/TEST SPLIT ===============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ============================================================
# ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ:LOGISTIC REGRESSION (baseline)
# ------------------------------------------------------------
# DT / RF: raw
# LR baseline: scaled αντίγραφα
# LR grid: scaling μέσα στο Pipeline (σωστό για CV)
# ============================================================
scaler = StandardScaler()
X_train_lr = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_lr  = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

# =============== CV ===============
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =============== HELPERS ===============
def save_confusion_matrix(y_true, y_pred, title, path_png):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign (0)", "Malignant (1)"])
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

def save_roc(y_true, y_proba, title, path_png):
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

def save_pr(y_true, y_proba, title, path_png):
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

def save_learning_curve(estimator, name, X_tr, y_tr, path_png):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_tr, y_tr, cv=cv, scoring="accuracy",
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8)
    )
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train Accuracy")
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="CV Accuracy")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve - {name}")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()

def tune_threshold_by_f1(y_true, y_proba):
    """Επιστρέφει threshold που μεγιστοποιεί F1 στο test set."""
    ps, rs, ths = precision_recall_curve(y_true, y_proba)
    if len(ths) != len(ps) - 1:
        ths = ths[:len(ps)-1]
    f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in ths]
    if len(f1s) == 0:
        return 0.5, f1_score(y_true, (y_proba >= 0.5).astype(int))
    best_i = int(np.argmax(f1s))
    return float(ths[best_i]), float(f1s[best_i])

def save_rf_permutation_importance(model, X_te, y_te, prefix):
    """
    Permutation importance για RandomForest (ROC-AUC).
    Χρησιμοποιεί scoring="roc_auc" για συμβατότητα με scikit-learn εκδόσεις.
    """
    perm = permutation_importance(
        model, X_te, y_te,
        scoring="roc_auc",
        n_repeats=30,
        random_state=42,
        n_jobs=-1
    )
    df_perm = (
        pd.DataFrame({
            "feature": feature_names,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std
        })
        .sort_values("perm_importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    df_perm.to_csv(os.path.join(OUT_DIR, f"{prefix}_permutation_importance.csv"), index=False)

def evaluate_and_save(
    name,
    fitted_estimator,
    is_grid=False,
    extra_prefix="",
    X_test_override=None,
    X_train_override=None,
    save_plots=True,
    save_model=True,
    save_threshold_cm=True,
    do_rf_permutation=False
):
    """
    Αξιολόγηση, plots, αποθήκευση μοντέλου και report.
    - Default: βγάζει ΟΛΑ.
    - Για baseline LR: θα το καλέσουμε με save_plots=False, save_model=False, save_threshold_cm=False (μόνο txt).
    """
    model = fitted_estimator.best_estimator_ if is_grid else fitted_estimator

    X_te = X_test_override if X_test_override is not None else X_test
    X_tr = X_train_override if X_train_override is not None else X_train

    # Προβλέψεις
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    rep = classification_report(y_test, y_pred, digits=3)

    # Report (πάντα)
    report_path = os.path.join(OUT_DIR, f"{extra_prefix}{name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(rep + f"\nAccuracy: {acc:.4f}\nROC-AUC: {auc:.4f}\nPR-AUC: {ap:.4f}\n")
        if is_grid:
            f.write(f"\nBest CV ROC-AUC: {fitted_estimator.best_score_:.4f}\n")
            f.write(f"Best Params: {fitted_estimator.best_params_}\n")

    print(f"\n=== {name} ===")
    print(rep)
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}")
    if is_grid:
        print(f"Best CV ROC-AUC: {fitted_estimator.best_score_:.4f}")
        print(f"Best Params: {fitted_estimator.best_params_}")

    # Threshold tuning report (txt πάντα)
    best_t, best_f1 = tune_threshold_by_f1(y_test, y_proba)
    y_pred_tuned = (y_proba >= best_t).astype(int)
    rep_tuned = classification_report(y_test, y_pred_tuned, digits=3)
    with open(os.path.join(OUT_DIR, f"{extra_prefix}{name}_report_threshold_tuned.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best threshold for F1: {best_t:.3f} | F1: {best_f1:.3f}\n\n")
        f.write(rep_tuned)

    # Plots (μόνο αν ζητηθεί)
    if save_plots:
        save_confusion_matrix(
            y_test, y_pred, f"{name} - Confusion Matrix",
            os.path.join(OUT_DIR, f"{extra_prefix}{name}_ConfusionMatrix.png")
        )
        save_roc(
            y_test, y_proba, f"{name} - ROC Curve",
            os.path.join(OUT_DIR, f"{extra_prefix}{name}_ROC.png")
        )
        save_pr(
            y_test, y_proba, f"{name} - Precision-Recall Curve",
            os.path.join(OUT_DIR, f"{extra_prefix}{name}_PR.png")
        )

        from sklearn.base import clone
        save_learning_curve(
            clone(model), name, X_tr, y_train,
            os.path.join(OUT_DIR, f"{extra_prefix}{name}_LearningCurve.png")
        )

    # Confusion matrix tuned (μόνο αν ζητηθεί)
    if save_threshold_cm and save_plots:
        save_confusion_matrix(
            y_test, y_pred_tuned, f"{name} - Confusion Matrix (Tuned @ {best_t:.3f})",
            os.path.join(OUT_DIR, f"{extra_prefix}{name}_ConfusionMatrix_TUNED.png")
        )

    # Αποθήκευση μοντέλου (μόνο αν ζητηθεί)
    if save_model:
        dump(model, os.path.join(OUT_DIR, f"{extra_prefix}{name}_best_model.joblib"))

    # RF importances
    if isinstance(model, RandomForestClassifier):
        # built-in importances
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        fi.to_csv(os.path.join(OUT_DIR, f"{extra_prefix}{name}_feature_importances.csv"), index=False)

        # permutation importance (επιπλέον)
        if do_rf_permutation:
            save_rf_permutation_importance(model, X_te, y_test, f"{extra_prefix}{name}")

# =============== ΜΟΝΤΕΛΑ ===============

# 1) Logistic Regression BASELINE (scaled, ΧΩΡΙΣ grid)
log_reg = LogisticRegression(max_iter=5000, solver="liblinear")

# 1b) Logistic Regression GRID (με scaling μέσα στο Pipeline)
log_reg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=5000, solver="liblinear"))
])
log_reg_params = {
    "logreg__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "logreg__penalty": ["l1", "l2"]
}
log_reg_grid = GridSearchCV(
    log_reg_pipe, log_reg_params, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
)

# 2) Decision Tree (GridSearchCV) - RAW
dt = DecisionTreeClassifier(random_state=42)
dt_params = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3, 5, 7, 9, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": [None, "sqrt", "log2"]
}
dt_grid = GridSearchCV(dt, dt_params, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)

# 3) Random Forest (GridSearchCV) - RAW
rf = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)

# =============== ΕΚΠΑΙΔΕΥΣΗ & ΑΞΙΟΛΟΓΗΣΗ ===============

# 1) Logistic Regression BASELINE: ΜΟΝΟ TXT (χωρίς γραφήματα & χωρίς joblib)
log_reg.fit(X_train_lr, y_train)
evaluate_and_save(
    "LogisticRegression_BASELINE",
    log_reg,
    is_grid=False,
    X_test_override=X_test_lr,
    X_train_override=X_train_lr,
    save_plots=False,
    save_model=False,
    save_threshold_cm=False
)

# 1b) Logistic Regression GRID
log_reg_grid.fit(X_train, y_train)  # raw X_train, pipeline κάνει scaling
evaluate_and_save(
    "LogisticRegression_TUNED",
    log_reg_grid,
    is_grid=True,
    X_test_override=X_test,     # raw test, pipeline κάνει scaling
    X_train_override=X_train,   # raw train, pipeline κάνει scaling
    save_plots=True,
    save_model=True,
    save_threshold_cm=True
)

# 2) Decision Tree (TUNED)
dt_grid.fit(X_train, y_train)
evaluate_and_save(
    "DecisionTree_TUNED",
    dt_grid,
    is_grid=True,
    save_plots=True,
    save_model=True,
    save_threshold_cm=True
)

# 3) Random Forest (TUNED)  + permutation importance
rf_grid.fit(X_train, y_train)
evaluate_and_save(
    "RandomForest_TUNED",
    rf_grid,
    is_grid=True,
    save_plots=True,
    save_model=True,
    save_threshold_cm=True,
    do_rf_permutation=True
)

print(f"\n✅ Όλα αποθηκεύτηκαν στο:\n{OUT_DIR}\n✔ Ολοκληρώθηκε.")





