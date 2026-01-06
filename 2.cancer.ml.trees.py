import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ============================================================
# BASE PATH
# ============================================================
BASE_DIR = r"C:\cancer.ml.py.knime"

# ============================================================
# ΦΑΚΕΛΟΙ
# ============================================================
MODEL_DIR = os.path.join(BASE_DIR, "outputs_full")      # ΕΔΩ ΕΙΝΑΙ ΤΑ joblib
PLOTS_DIR = os.path.join(BASE_DIR, "outputfulltrees")   # ΕΔΩ ΘΑ ΜΠΟΥΝ TA PNG
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# FEATURE NAMES ΑΠΟ data.csv
# ============================================================
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
df = pd.read_csv(DATA_PATH)

if "id" in df.columns:
    df = df.drop(columns=["id"])

assert "diagnosis" in df.columns, "Λείπει η στήλη 'diagnosis'."
feature_names = df.drop(columns=["diagnosis"]).columns.tolist()

# ============================================================
# 1) DECISION TREE
# ============================================================
dt_path = os.path.join(MODEL_DIR, "DecisionTree_TUNED_best_model.joblib")
dt = joblib.load(dt_path)

# πλήρες δέντρο
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(
    dt,
    filled=True,
    feature_names=feature_names,
    class_names=["Benign", "Malignant"],
    fontsize=8
)
ax.set_title("DecisionTree_TUNED - Full")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "DecisionTree_TUNED_full.png"), dpi=200)
plt.close(fig)

# περιορισμένο βάθος
fig, ax = plt.subplots(figsize=(16, 9))
plot_tree(
    dt,
    max_depth=3,
    filled=True,
    feature_names=feature_names,
    class_names=["Benign", "Malignant"],
    fontsize=9
)
ax.set_title("DecisionTree_TUNED - max_depth=3")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "DecisionTree_TUNED_depth3.png"), dpi=200)
plt.close(fig)

# ============================================================
# 2) RANDOM FOREST – 3 ΕΠΙΜΕΡΟΥΣ ΔΕΝΤΡΑ
# ============================================================
rf_path = os.path.join(MODEL_DIR, "RandomForest_TUNED_best_model.joblib")
rf = joblib.load(rf_path)

for i in range(min(3, len(rf.estimators_))):
    est = rf.estimators_[i]
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(
        est,
        max_depth=3,
        filled=True,
        feature_names=feature_names,
        class_names=["Benign", "Malignant"],
        fontsize=8
    )
    ax.set_title(f"RandomForest_TUNED - Tree #{i} (max_depth=3)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOTS_DIR, f"RandomForest_TUNED_tree{i}_depth3.png"),
        dpi=200
    )
    plt.close(fig)

print("✅ Τα trees αποθηκεύτηκαν στο:", PLOTS_DIR)

