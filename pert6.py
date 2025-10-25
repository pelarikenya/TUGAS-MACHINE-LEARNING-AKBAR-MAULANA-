# rf_pipeline_pertemuan6.py
# Python 3.10
# Lembar Kerja Pertemuan 6 — Random Forest untuk Klasifikasi
# =========================================================

import os
import sys
import warnings
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

# -----------------------
# Config / reproducibility
# -----------------------
SEED = 42
RANDOM_STATE = SEED
np.random.seed(SEED)

# -----------------------
# Helper functions
# -----------------------
def try_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        return None

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

# -----------------------
# Langkah 0/1: Muat Data
# - Pilihan A: processed_kelulusan.csv
# - Pilihan B: X_train.csv / X_val.csv / X_test.csv + y_*.csv
# Jika tidak ditemukan -> error message
# -----------------------
print("=== LANGKAH 0/1: MEMUAT DATA ===")

# Look for processed file first
df = try_read_csv("processed_kelulusan.csv")

if df is not None:
    print("Sumber: processed_kelulusan.csv (pilihan A)")
    # Ensure target 'Lulus' exists
    if "Lulus" not in df.columns:
        raise ValueError("File processed_kelulusan.csv tidak memiliki kolom 'Lulus'.")
    X = df.drop("Lulus", axis=1)
    y = df["Lulus"]
else:
    # Try to read split files
    X_train_f = try_read_csv("X_train.csv")
    X_val_f   = try_read_csv("X_val.csv")
    X_test_f  = try_read_csv("X_test.csv")
    y_train_f = try_read_csv("y_train.csv")
    y_val_f   = try_read_csv("y_val.csv")
    y_test_f  = try_read_csv("y_test.csv")

    if X_train_f is not None and X_val_f is not None and X_test_f is not None \
       and y_train_f is not None and y_val_f is not None and y_test_f is not None:
        print("Sumber: file split tersedia (pilihan B). Memuat langsung X_train/X_val/X_test dan y_*.")
        # squeeze y if they were saved as single-column CSVs
        X_train = X_train_f.copy()
        X_val   = X_val_f.copy()
        X_test  = X_test_f.copy()
        y_train = y_train_f.squeeze() if isinstance(y_train_f, pd.DataFrame) else y_train_f
        y_val   = y_val_f.squeeze() if isinstance(y_val_f, pd.DataFrame) else y_val_f
        y_test  = y_test_f.squeeze() if isinstance(y_test_f, pd.DataFrame) else y_test_f

        # Make sure y dtype is integer
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)

        print("Shapes (loaded):", X_train.shape, X_val.shape, X_test.shape)
    else:
        raise FileNotFoundError(
            "Tidak menemukan 'processed_kelulusan.csv' maupun semua file split (X_train/X_val/X_test & y_*.csv).\n"
            "Silakan tempatkan salah satu sumber data di folder kerja."
        )

# If we loaded processed file, do the split here
if df is not None:
    print("Melakukan stratified split 70/15/15 dari processed_kelulusan.csv ...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print("Shapes (split):", X_train.shape, X_val.shape, X_test.shape)

# Basic checks
assert set([0,1]).issuperset(set(y_train.unique())), "Label y harus berisi 0/1"
print("Label unik (train):", sorted(y_train.unique()))

# -----------------------
# Langkah 2: Pipeline & Baseline Random Forest
# -----------------------
print("\n=== LANGKAH 2: PIPELINE & BASELINE RANDOM FOREST ===")

# select numeric columns (all features expected numeric)
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
print("Kolom numerik:", num_cols)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", rf)
])

# Fit baseline
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipe.fit(X_train, y_train)

y_val_pred = pipe.predict(X_val)
print("Baseline RF — F1(val, macro):", f1_score(y_val, y_val_pred, average="macro"))
print("Baseline classification report (val):")
print(classification_report(y_val, y_val_pred, digits=3))

# -----------------------
# Langkah 3: Validasi Silang (CV)
# -----------------------
print("\n=== LANGKAH 3: VALIDASI SILANG (StratifiedKFold) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro (train): %.4f ± %.4f" % (cv_scores.mean(), cv_scores.std()))

# -----------------------
# Langkah 4: Tuning Ringkas (GridSearchCV)
# -----------------------
print("\n=== LANGKAH 4: GRID SEARCH TUNING RINGKAS ===")
param_grid = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param_grid, cv=skf, scoring="f1_macro",
                  n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
print("Best CV F1 (train):", gs.best_score_)

best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("Best RF — F1(val, macro):", f1_score(y_val, y_val_best, average="macro"))
print("Classification report (best, val):")
print(classification_report(y_val, y_val_best, digits=3))

# -----------------------
# Langkah 5: Evaluasi Akhir (Test Set)
# -----------------------
print("\n=== LANGKAH 5: EVALUASI AKHIR (TEST SET) ===")
final_model = best_model  # gunakan model terbaik dari gridsearch

y_test_pred = final_model.predict(X_test)
f1_test = f1_score(y_test, y_test_pred, average="macro")
print("F1(test, macro):", f1_test)
print("Classification report (test):")
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# ROC-AUC & PR curves (jika predict_proba ada)
ensure_dir("outputs")
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        auc_test = roc_auc_score(y_test, y_test_proba)
        print("ROC-AUC(test):", auc_test)
    except Exception as e:
        print("Tidak dapat menghitung ROC-AUC:", e)
        auc_test = None

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_test:.3f})" if auc_test is not None else "ROC")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/roc_test.png", dpi=120)
    plt.close()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Test Set")
    plt.tight_layout()
    plt.savefig("outputs/pr_test.png", dpi=120)
    plt.close()
else:
    print("Model tidak memiliki method predict_proba; ROC & PR tidak tersedia.")

# -----------------------
# Langkah 6: Pentingnya Fitur (Feature Importance)
# -----------------------
print("\n=== LANGKAH 6: FEATURE IMPORTANCE ===")
try:
    # try to get feature names after preprocessing
    try:
        feat_names = final_model.named_steps["pre"].get_feature_names_out()
    except Exception:
        # Fallback: use numeric column names
        feat_names = np.array(num_cols)

    importances = final_model.named_steps["clf"].feature_importances_
    fi = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    top_n = fi[:20]
    print("Top feature importances:")
    for name, val in top_n:
        print(f"{name}: {val:.4f}")

    # Save to CSV
    fi_df = pd.DataFrame(fi, columns=["feature", "importance"]).sort_values("importance", ascending=False)
    fi_df.to_csv("outputs/feature_importance.csv", index=False)
    print("Feature importance disimpan di outputs/feature_importance.csv")
except Exception as e:
    print("Feature importance tidak tersedia / terjadi error:", e)

# -----------------------
# Langkah 7: Simpan Model
# -----------------------
print("\n=== LANGKAH 7: SIMPAN MODEL ===")
joblib.dump(final_model, "rf_model.pkl")
print("Model disimpan sebagai rf_model.pkl")

# -----------------------
# Langkah 8: Cek Inference Lokal (contoh)
# -----------------------
print("\n=== LANGKAH 8: CEK INFERENCE LOKAL (CONTOH) ===")
try:
    mdl = joblib.load("rf_model.pkl")
    sample = pd.DataFrame([{
        # gunakan nama kolom persis seperti X_train
        **{c: float(X_train[c].median()) for c in X_train.columns}
    }])
    pred = mdl.predict(sample)[0]
    proba = None
    if hasattr(mdl, "predict_proba"):
        proba = float(mdl.predict_proba(sample)[:, 1][0])
    print("Contoh prediksi pada sample (median fitur):", int(pred), "proba:", proba)
except Exception as e:
    print("Gagal melakukan inference contoh:", e)

# -----------------------
# Ringkasan / Checklist (terminal)
# -----------------------
print("\n=== RINGKASAN AKHIR ===")
print("- Baseline RF + tuning (GridSearchCV) telah dilakukan.")
print(f"- Best CV F1 (train): {gs.best_score_:.4f}")
print(f"- F1 (test, macro): {f1_test:.4f}")
print("- Files output: outputs/roc_test.png, outputs/pr_test.png, outputs/feature_importance.csv, rf_model.pkl")
print("- Selesai.")
