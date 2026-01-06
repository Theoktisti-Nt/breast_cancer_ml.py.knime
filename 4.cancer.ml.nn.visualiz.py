# Ερμηνεύσιμο γράφημα επιρροής χαρακτηριστικών NN
# Χρησιμοποιεί το NN_best_model.joblib από scikit-learn MLP
# ποιες είσοδοι έχουν ισχυρότερη επιρροή στην απόφαση (benign vs malignant)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import joblib

# --- Paths ---
BASE_DIR  = r"C:\cancer.ml.py.knime"
MODEL_P   = os.path.join(BASE_DIR, r"outputfullnn\NN_best_model.joblib")
DATA_P    = os.path.join(BASE_DIR, "data.csv")
OUT_PNG   = os.path.join(BASE_DIR, r"outputfulnnvisulaz\NN_feature_influence.png")

# --- Load model & data ---
model = joblib.load(MODEL_P)
mlp = getattr(model, "named_steps", {}).get("mlp", model)

coefs = mlp.coefs_
assert len(coefs) >= 2, "Το δίκτυο πρέπει να έχει τουλάχιστον 1 hidden layer"

# --- Read feature names ---
df = pd.read_csv(DATA_P)
if "id" in df.columns:
    df = df.drop(columns=["id"])
feature_names = [c for c in df.columns if c != "diagnosis"]

# --- Compute cumulative influence from Input -> Output ---
W_total = np.abs(coefs[0]) @ np.abs(coefs[1])
for W in coefs[2:]:
    W_total = W_total @ np.abs(W)

# Signed influence: multiply all weights to output and average per input feature
signed_influence = coefs[0]
for W in coefs[1:]:
    signed_influence = signed_influence @ W

final_influence = signed_influence.mean(axis=1)  # mean effect per input feature

# --- Sort top influences ---
sorted_idx = np.argsort(np.abs(final_influence))[::-1]
top_n = 15
top_idx = sorted_idx[:top_n]
top_features = [feature_names[i] for i in top_idx]
top_values = final_influence[top_idx]

# --- Plot ---
plt.figure(figsize=(12, 7))
norm = TwoSlopeNorm(vmin=final_influence.min(), vcenter=0, vmax=final_influence.max())
colors = plt.cm.coolwarm(norm(top_values))
bars = plt.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
plt.title("Top 15 Feature Influences in Neural Network\n(Positive → Malignant, Negative → Benign)", fontsize=14)
plt.xlabel("Influence weight (aggregated through network)")
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=220)
plt.close()
print(f"✅ Saved: {OUT_PNG}")



# STRING-style δίκτυο: κόμβοι=features, ακμές=“συνεργασία” στο NN
# Φορτώνει scikit-learn MLP (joblib) και data.csv για labels

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import joblib

# ---------- ΡΥΘΜΙΣΕΙΣ ΔΙΑΔΡΟΜΩΝ ----------
BASE = r"C:\cancer.ml.py.knime"
MODEL = os.path.join(BASE, r"outputfullnn\NN_best_model.joblib")
CSV   = os.path.join(BASE, "data.csv")
OUT   = os.path.join(BASE, r"outputfulnnvisulaz\NN_STRING_style.png")
OUT_SVG = os.path.join(BASE, r"outputfulnnvisulaz\NN_STRING_style.svg")

# ---------- ΦΟΡΤΩΣΗ ΜΟΝΤΕΛΟΥ & FEATURES ----------
pipe = joblib.load(MODEL)
mlp = getattr(pipe, "named_steps", {}).get("mlp", pipe)  # πιάσε τον MLPClassifier
W = mlp.coefs_            # [W0: nF×H1, W1: H1×H2, W2: H2×1] (ή παραπάνω hidden)
assert len(W) >= 2, "Το NN χρειάζεται ≥1 hidden layer"

df = pd.read_csv(CSV)
if "id" in df.columns:
    df = df.drop(columns=["id"])
features = [c for c in df.columns if c != "diagnosis"]
nF = len(features)

# ---------- ΔΙΑΝΥΣΜΑ “ΔΙΑΔΡΟΜΗΣ” ΚΑΘΕ FEATURE ΠΡΟΣ ΤΗΝ ΕΞΟΔΟ ----------
# Για κάθε feature i φτιάχνουμε ένα vector συνεισφοράς προς τους τελευταίους κρυφούς νευρώνες,
# σταθμισμένο με τα βάρη προς την έξοδο. Έτσι συγκρίνουμε “μονοπάτια” (όχι μόνο ένα scalar).
A = W[0]                        # nF × H1
# προώθησε μέχρι το τελευταίο hidden
Z = A
for L in range(1, len(W)-1):
    Z = Z @ W[L]               # nF × H_last

w_out = W[-1].reshape(-1)      # H_last
# συνεισφορά feature i ανά νευρώνα τελευταίου hidden, σταθμισμένη με προς-έξοδο βάρος
contrib = Z * w_out            # nF × H_last

# ---------- ΟΜΟΙΟΤΗΤΑ ΖΕΥΓΩΝ FEATURES (STRING-style edges) ----------
# Χρησιμοποιούμε cosine similarity στα “μονοπάτια” contrib.
norms = np.linalg.norm(contrib, axis=1, keepdims=True) + 1e-12
U = contrib / norms            # unit vectors
S = U @ U.T                    # nF × nF, στο [-1, 1]; διαγώνιος=1

# ---------- ΕΠΙΛΟΓΗ ΑΚΜΩΝ ----------
KEEP_TOP = 0.15   # κράτα top 15% σε |similarity| (STRING-style αραιό δίκτυο)
absS = np.abs(S.copy())
# αγνόησε διαγώνιο & κάτω τρίγωνο (μη διπλές ακμές)
mask = np.triu(np.ones_like(absS, dtype=bool), k=1)
vals = absS[mask]
thr = np.quantile(vals, 1.0 - KEEP_TOP) if vals.size else 1.0

edges = []
for i in range(nF):
    for j in range(i+1, nF):
        sim = S[i, j]
        if abs(sim) >= thr:
            edges.append((i, j, sim))

# ---------- ΔΙΑΤΑΞΗ ΚΟΜΒΩΝ (ΚΥΚΛΙΚΗ) ----------
R = 8.5
angles = np.linspace(0, 2*np.pi, nF, endpoint=False)
coords = np.c_[R*np.cos(angles), R*np.sin(angles)]

# ---------- ΣΧΕΔΙΑΣΗ ----------
plt.figure(figsize=(16, 16))
ax = plt.gca(); ax.axis('off'); ax.set_aspect('equal')

# Χρώμα ακμών: μπλε (αρνητική συνεργασία) / κόκκινο (θετική)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cmap = plt.get_cmap("coolwarm")

# πάχος γραμμών ∝ |similarity|
if edges:
    max_abs = max(abs(w) for *_, w in edges)
else:
    max_abs = 1.0

for i, j, s in edges:
    x1, y1 = coords[i]; x2, y2 = coords[j]
    lw = 0.8 + 6.0 * (abs(s) / max_abs)
    col = cmap(norm(s))
    ax.plot([x1, x2], [y1, y2], color=col, linewidth=lw, alpha=0.55, zorder=1)

# Κόμβοι
ax.scatter(coords[:,0], coords[:,1], s=280, facecolor="white",
           edgecolor="black", linewidth=1.8, zorder=3)

# Labels features
for k, (x, y) in enumerate(coords):
    ax.text(x, y, features[k], ha='center', va='center',
            fontsize=9, color='black', zorder=4)

# Τίτλος/Υπόμνημα
plt.suptitle(
    "STRING-style Feature Influence Network (Neural Network)\n"
    "Κόμβοι: χαρακτηριστικά • Ακμές: ομοιότητα «μονοπατιών» προς την έξοδο (cosine)\n"
    "Χρώμα: πρόσημο συνεργασίας (μπλε −, κόκκινο +) • Πάχος: |ομοιότητα| • Top 15% ακμών",
    y=0.94, fontsize=13)

plt.tight_layout(rect=[0,0,1,0.93])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=220)
plt.savefig(OUT_SVG)   # για άψογη ποιότητα σε thesis
plt.close()
print("✅ Saved:", OUT)
print("✅ Saved:", OUT_SVG)



# STRING-style δίκτυο με clustering (mean / se / worst)
# pip install joblib numpy pandas matplotlib

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import joblib

# ----- Paths -----
BASE = r"C:\cancer.ml.py.knime"
MODEL = os.path.join(BASE, r"outputfullnn\NN_best_model.joblib")
CSV   = os.path.join(BASE, "data.csv")
OUT   = os.path.join(BASE, r"outputfulnnvisulaz\NN_STRING_clusters.png")

# ----- Load model -----
pipe = joblib.load(MODEL)
mlp = getattr(pipe, "named_steps", {}).get("mlp", pipe)
W = mlp.coefs_
df = pd.read_csv(CSV)
if "id" in df.columns:
    df = df.drop(columns=["id"])
features = [c for c in df.columns if c != "diagnosis"]
nF = len(features)

# ----- Propagate weights -----
A = W[0]
Z = A
for L in range(1, len(W)-1):
    Z = Z @ W[L]
w_out = W[-1].reshape(-1)
contrib = Z * w_out

# ----- Cosine similarity -----
norms = np.linalg.norm(contrib, axis=1, keepdims=True) + 1e-12
U = contrib / norms
S = U @ U.T

KEEP_TOP = 0.15
mask = np.triu(np.ones_like(S, dtype=bool), k=1)
vals = np.abs(S[mask])
thr = np.quantile(vals, 1.0 - KEEP_TOP) if vals.size else 1.0

edges = [(i, j, S[i,j]) for i in range(nF) for j in range(i+1, nF) if abs(S[i,j]) >= thr]

# ----- Cluster group assignment -----
def feature_group(name):
    if "_mean" in name:
        return "mean"
    elif "_se" in name:
        return "se"
    elif "_worst" in name:
        return "worst"
    else:
        return "other"

groups = [feature_group(f) for f in features]
colors_group = {"mean": "#66c2a5", "se": "#8da0cb", "worst": "#fc8d62", "other": "#a6d854"}

# ----- Layout: circular per cluster -----
clusters = {"mean": [], "se": [], "worst": [], "other": []}
for i, g in enumerate(groups):
    clusters[g].append(i)

angles = {}
start_angle = 0
for g, idxs in clusters.items():
    n = len(idxs)
    if n == 0:
        continue
    a = np.linspace(start_angle, start_angle + 2*np.pi * (n/nF), n, endpoint=False)
    for i, ang in zip(idxs, a):
        angles[i] = ang
    start_angle += 2*np.pi * (n/nF)

R = 9
coords = np.array([[R*np.cos(angles[i]), R*np.sin(angles[i])] for i in range(nF)])

# ----- Plot -----
plt.figure(figsize=(16,16))
ax = plt.gca()
ax.axis("off")
ax.set_aspect("equal")

norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cmap = plt.get_cmap("coolwarm")

# Draw edges
if edges:
    max_abs = max(abs(w) for *_, w in edges)
else:
    max_abs = 1.0

for i,j,s in edges:
    x1,y1 = coords[i]; x2,y2 = coords[j]
    lw = 0.8 + 5*(abs(s)/max_abs)
    col = cmap(norm(s))
    ax.plot([x1,x2],[y1,y2], color=col, linewidth=lw, alpha=0.55)

# Draw nodes
for i,(x,y) in enumerate(coords):
    g = groups[i]
    fc = colors_group.get(g,"#ccc")
    ax.scatter(x, y, s=340, color=fc, edgecolor="black", linewidth=1.6, zorder=3)
    ax.text(x, y, features[i], ha="center", va="center", fontsize=8, zorder=4)

# Legend
for label, color in colors_group.items():
    plt.scatter([], [], c=color, label=label, s=180, edgecolor="black", linewidth=1)
plt.legend(title="Cluster", loc="upper left")

plt.suptitle("STRING-style Modular Feature Influence Network (Neural Network)\n"
             "Χρώμα κόμβου: ομάδα (mean / se / worst) • Χρώμα ακμής: πρόσημο συνεργασίας (μπλε −, κόκκινο +)\n"
             "Πάχος: ισχύς σύνδεσης (|cosine|) • Top 15% ακμών",
             fontsize=13, y=0.94)

plt.tight_layout(rect=[0,0,1,0.93])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=220)
plt.close()
print("✅ Saved:", OUT)


