# ===============================================================
# NESTED CROSS-VALIDATION + BOOTSTRAP
# ===============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


# -----------------------------
# ΡΥΘΜΙΣΕΙΣ (ΝΕΕΣ ΔΙΑΔΡΟΜΕΣ)
# -----------------------------
BASE_DIR   = r"C:\cancer.ml.py.knime"
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")



OUT_DIR     = os.path.join(BASE_DIR, "outputfullnested")
RESULTS_DIR = Path(OUT_DIR) / "nested_cv_full"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ
# -----------------------------
df = pd.read_csv(DATA_PATH)

if "id" in df.columns:
    df = df.drop(columns=["id"])

if "diagnosis" not in df.columns:
    raise ValueError("Λείπει η στήλη 'diagnosis' από το data.csv")

le = LabelEncoder()
y = le.fit_transform(df["diagnosis"])   # π.χ. B=0, M=1
X = df.drop(columns=["diagnosis"])

print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")
print(f"✅ Outputs: {RESULTS_DIR}")


# ======================================================
# CUSTOM SCORERS
# ======================================================
def f1_scorer_threshold(threshold):
    """Custom scorer function που δουλεύει απευθείας με estimator."""
    def scorer(estimator, X, y_true):
        try:
            y_proba = estimator.predict_proba(X)[:, 1]
        except AttributeError:
            # Αν δεν υπάρχει predict_proba (π.χ. SVM)
            y_score = estimator.decision_function(X)
            # normalize to [0,1]
            denom = (y_score.max() - y_score.min())
            if denom == 0:
                y_proba = np.zeros_like(y_score, dtype=float)
            else:
                y_proba = (y_score - y_score.min()) / denom

        y_pred = (y_proba >= threshold).astype(int)
        return f1_score(y_true, y_pred, zero_division=0)
    return scorer


# Ορισμός scorers
f1_050 = f1_scorer_threshold(0.50)
f1_040 = f1_scorer_threshold(0.40)

SCORERS = [
    ("roc_auc", "roc_auc"),  # threshold-free metric
    ("f1_050",  f1_050),     # F1 @ 0.50
    ("f1_040",  f1_040),     # F1 @ 0.40
]


# ======================================================
# NESTED CV + BOOTSTRAP
# ======================================================
def nested_cv_bootstrap(
    name, estimator, param_grid,
    X, y,
    scorers=SCORERS,
    outer_splits=3, inner_splits=2,
    n_boot=10, random_state=42
):
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    for metric_label, scoring in scorers:
        print(f"\n=== {name} — Nested CV ({metric_label}) ===")

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )

        outer_scores = cross_val_score(
            grid, X, y,
            cv=outer_cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise"
        )

        mean_outer = float(np.mean(outer_scores))
        std_outer  = float(np.std(outer_scores))
        print(f"Nested CV {metric_label}: {mean_outer:.4f} ± {std_outer:.4f}")

        # Bootstrap (σε resampled datasets) -> CI για best inner-CV score
        boot_scores = []
        for i in range(n_boot):
            X_b, y_b = resample(X, y, random_state=random_state + i)

            grid.fit(X_b, y_b)
            boot_scores.append(float(grid.best_score_))

        ci_low, ci_high = np.percentile(boot_scores, [2.5, 97.5])
        print(f"Bootstrap {metric_label} 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        out_file = RESULTS_DIR / f"{name.replace(' ', '_')}_{metric_label}_nestedCV.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"{name} — Nested CV ({metric_label})\n")
            f.write(f"Nested CV {metric_label}: {mean_outer:.6f} ± {std_outer:.6f}\n")
            f.write(f"Bootstrap 95% CI: [{ci_low:.6f}, {ci_high:.6f}]\n")

        print(f"📝 Saved: {out_file}")


# ======================================================
# ΟΡΙΣΜΟΣ ΜΟΝΤΕΛΩΝ
# ======================================================
models = []

# Logistic Regression
logreg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, solver="liblinear"))
])
logreg_grid = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__penalty": ["l1", "l2"]
}
models.append(("Logistic Regression", logreg_pipe, logreg_grid))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": [None, "sqrt", "log2"]
}
models.append(("Decision Tree", dt, dt_grid))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf_grid = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}
models.append(("Random Forest", rf, rf_grid))

# MLP
mlp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        activation="relu",
        max_iter=400,
        random_state=42,
        early_stopping=True
    ))
])

mlp_grid = {
    "clf__hidden_layer_sizes": [(64, 32), (128, 64)],
    "clf__alpha": [1e-4, 1e-3],
    "clf__learning_rate_init": [1e-3, 5e-4]
}
models.append(("Neural Network (MLP)", mlp_pipe, mlp_grid))


# ======================================================
# ΕΚΤΕΛΕΣΗ (FAST SETTINGS)
# ======================================================
for name, est, grid in models:
    nested_cv_bootstrap(
        name=name,
        estimator=est,
        param_grid=grid,
        X=X, y=y,
        scorers=SCORERS,
        outer_splits=3,
        inner_splits=2,
        n_boot=10,
        random_state=42
    )

print("\n✅ Nested CV + Bootstrap ολοκληρώθηκε.")
print(f"📁 Αποτελέσματα: {RESULTS_DIR}")
