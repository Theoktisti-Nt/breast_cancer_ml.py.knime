# nn_all_in_one
# pip install scikit-learn pandas numpy matplotlib joblib

# --- ΜΗ ΔΙΑΔΡΑΣΤΙΚΟ BACKEND ΓΙΑ PLOTS (χωρίς tkinter) ---
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

# --- IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
    precision_recall_curve, f1_score
)
from sklearn.inspection import permutation_importance

# =============== ΡΥΘΜΙΣΕΙΣ ΔΙΑΔΡΟΜΩΝ ===============
BASE_DIR  = r"C:\cancer.ml.py.knime"
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
OUT_DIR   = os.path.join(BASE_DIR, "outputfullnn")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =============== LOAD DATA ===============
df = pd.read_csv(DATA_PATH)
assert "diagnosis" in df.columns, "Λείπει η στήλη 'diagnosis' (B/M)."
if "id" in df.columns:
    df = df.drop(columns=["id"])

le = LabelEncoder()
y = le.fit_transform(df["diagnosis"])  # B=0, M=1
X = df.drop(columns=["diagnosis"])
feature_names = X.columns.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# =============== NN PIPELINE ===============
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=1e-3,                 # L2
        learning_rate_init=0.001,
        early_stopping=True,
        n_iter_no_change=20,
        max_iter=2000,
        random_state=42
    ))
])

#  do_grid
do_grid = True
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if do_grid:
    param_grid = {
        "mlp__hidden_layer_sizes": [(32, 16), (64, 32), (32,)],
        "mlp__alpha": [1e-2, 1e-3, 1e-4]
    }
    grid = GridSearchCV(
        pipe, param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_tr, y_tr)
    model = grid.best_estimator_
    best_info = f"Best params: {grid.best_params_} | CV ROC-AUC: {grid.best_score_:.4f}"
else:
    model = pipe.fit(X_tr, y_tr)
    best_info = "(no grid search)"

# =============== ΑΞΙΟΛΟΓΗΣΗ ===============
proba = model.predict_proba(X_te)[:, 1]
pred  = (proba >= 0.5).astype(int)

acc = accuracy_score(y_te, pred)
auc = roc_auc_score(y_te, proba)
ap  = average_precision_score(y_te, proba)
rep = classification_report(y_te, pred, digits=3)

print(rep)
print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}")
print(best_info)

# Save text report
with open(os.path.join(OUT_DIR, "NN_report.txt"), "w", encoding="utf-8") as f:
    f.write(rep + f"\nAccuracy: {acc:.4f}\nROC-AUC: {auc:.4f}\nPR-AUC: {ap:.4f}\n")
    f.write(best_info + "\n")

# Plots: ROC, PR, Confusion
RocCurveDisplay.from_predictions(y_te, proba)
plt.title("NN - ROC")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_ROC.png"), dpi=160)
plt.close()

PrecisionRecallDisplay.from_predictions(y_te, proba)
plt.title("NN - PR")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_PR.png"), dpi=160)
plt.close()

ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Benign (0)", "Malignant (1)"])
plt.title("NN - Confusion")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_Confusion.png"), dpi=160)
plt.close()

# =============== THRESHOLD TUNING (F1) ===============
def tune_threshold_by_f1(y_true, y_proba):
    ps, rs, ths = precision_recall_curve(y_true, y_proba)
    # ths length = len(ps)-1
    ths = ths[:len(ps)-1]
    f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in ths] if len(ths) else []
    if not f1s:
        return 0.5, f1_score(y_true, (y_proba >= 0.5).astype(int))
    bi = int(np.argmax(f1s))
    return float(ths[bi]), float(f1s[bi])

best_t, best_f1 = tune_threshold_by_f1(y_te, proba)
pred_tuned = (proba >= best_t).astype(int)
rep_tuned = classification_report(y_te, pred_tuned, digits=3)

with open(os.path.join(OUT_DIR, "NN_report_threshold_tuned.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best threshold for F1: {best_t:.3f} | F1: {best_f1:.3f}\n\n")
    f.write(rep_tuned)

ConfusionMatrixDisplay.from_predictions(y_te, pred_tuned, display_labels=["Benign (0)", "Malignant (1)"])
plt.title(f"NN - Confusion (Tuned @ {best_t:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_Confusion_TUNED.png"), dpi=160)
plt.close()

# =============== SAVE MODEL ===============
dump(model, os.path.join(OUT_DIR, "NN_best_model.joblib"))

# =============== HEATMAPS ΒΑΡΩΝ ===============
# Θα εξάγουμε τα βάρη από το τελικό MLP μέσα στο pipeline
mlp = model.named_steps["mlp"]
coefs = mlp.coefs_      # λίστα με W matrices: [W0(n_featuresxH1), W1(H1xH2), W2(H2x1)]
intercepts = mlp.intercepts_

W_input_hidden1 = coefs[0]          # shape: (n_features, 32)
W_hidden1_hidden2 = coefs[1]        # shape: (32, 16)
W_hidden2_out = coefs[2].reshape(-1)  # shape: (16,)

# --- Heatmap: Input→Hidden1 με labels χαρακτηριστικών
plt.figure(figsize=(12, 8))
im = plt.imshow(W_input_hidden1, aspect="auto", cmap="coolwarm")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names, fontsize=8)
plt.xticks(
    ticks=np.arange(W_input_hidden1.shape[1]),
    labels=[f"h1_{i}" for i in range(W_input_hidden1.shape[1])],
    fontsize=7
)
plt.title("NN Weights Heatmap: Input → Hidden1")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_Heatmap_Input_H1.png"), dpi=180)
plt.close()

# --- Heatmap: Hidden1→Hidden2
plt.figure(figsize=(10, 7))
im = plt.imshow(W_hidden1_hidden2, aspect="auto", cmap="coolwarm")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.yticks(
    ticks=np.arange(W_hidden1_hidden2.shape[0]),
    labels=[f"h1_{i}" for i in range(W_hidden1_hidden2.shape[0])],
    fontsize=7
)
plt.xticks(
    ticks=np.arange(W_hidden1_hidden2.shape[1]),
    labels=[f"h2_{j}" for j in range(W_hidden1_hidden2.shape[1])],
    fontsize=7
)
plt.title("NN Weights Heatmap: Hidden1 → Hidden2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_Heatmap_H1_H2.png"), dpi=180)
plt.close()

# --- Barplot: Hidden2→Output
plt.figure(figsize=(9, 4))
plt.bar(np.arange(len(W_hidden2_out)), W_hidden2_out)
plt.xlabel("Hidden2 neuron index")
plt.ylabel("Weight to output")
plt.title("NN Weights: Hidden2 → Output")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_Bar_H2_to_Output.png"), dpi=160)
plt.close()

# =============== FEATURE IMPORTANCE (2 τρόποι) ===============
# (A) Proxy από βάρη: μέση απόλυτη τιμή ανά feature στο Input→Hidden1
feat_proxy = pd.Series(
    np.mean(np.abs(W_input_hidden1), axis=1),
    index=feature_names
).sort_values(ascending=False)

feat_proxy.to_csv(os.path.join(OUT_DIR, "NN_feature_importance_proxy_from_weights.csv"))

plt.figure(figsize=(10, 6))
feat_proxy.head(15).iloc[::-1].plot(kind="barh")
plt.title("NN Feature 'Importance' (proxy from |weights|, Input→Hidden1)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_FeatureImportance_Proxy_Top15.png"), dpi=160)
plt.close()

# (B) Permutation Importance
perm = permutation_importance(
    model, X_te, y_te,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

feat_perm = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
feat_perm.to_csv(os.path.join(OUT_DIR, "NN_feature_importance_permutation.csv"))

plt.figure(figsize=(10, 6))
feat_perm.head(15).iloc[::-1].plot(kind="barh")
plt.title("NN Feature Importance (Permutation, metric=ROC-AUC)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "NN_FeatureImportance_Permutation_Top15.png"), dpi=160)
plt.close()

print("✅ Όλα αποθηκεύτηκαν στο:", OUT_DIR)
