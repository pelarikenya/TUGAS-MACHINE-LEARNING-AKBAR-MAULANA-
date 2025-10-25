# ================================================
# 📗 Lembar Kerja Pertemuan 5 — Modeling
# Topik: Selection • Training • Validation • Testing • Deployment (optional)
# ================================================

# ==== Langkah 0 — Impor Library ====
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import joblib

# =====================================================
# ==== Langkah 1 — Muat Data ====
# =====================================================
print("=== LANGKAH 1: MUAT DATA ===")

df = pd.read_csv("processed_kelulusan.csv")

# Pisahkan fitur dan target
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split data: train 70%, val 15%, test 15%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Bentuk data:", X_train.shape, X_val.shape, X_test.shape)

# =====================================================
# ==== Langkah 2 — Baseline Model (Logistic Regression) ====
# =====================================================
print("\n=== LANGKAH 2: BASELINE MODEL ===")

num_cols = X_train.select_dtypes(include="number").columns

# Preprocessing numerik
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

# Model Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

# Buat pipeline
pipe_lr = Pipeline([
    ("pre", pre),
    ("clf", logreg)
])

# Latih model baseline
pipe_lr.fit(X_train, y_train)

# Prediksi validasi
y_val_pred = pipe_lr.predict(X_val)

# Evaluasi
print("Baseline (LogReg) F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# =====================================================
# ==== Langkah 3 — Model Alternatif (Random Forest) ====
# =====================================================
print("\n=== LANGKAH 3: RANDOM FOREST ===")

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

pipe_rf = Pipeline([
    ("pre", pre),
    ("clf", rf)
])

pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)

print("RandomForest F1(val):", f1_score(y_val, y_val_rf, average="macro"))
print(classification_report(y_val, y_val_rf, digits=3))

# =====================================================
# ==== Langkah 4 — Validasi Silang & Tuning Ringkas ====
# =====================================================
print("\n=== LANGKAH 4: GRID SEARCH TUNING ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param = {
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(
    pipe_rf,
    param_grid=param,
    cv=skf,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

# Model terbaik
best_rf = gs.best_estimator_

# Evaluasi di validation set
y_val_best = best_rf.predict(X_val)
print("Best RF F1(val):", f1_score(y_val, y_val_best, average="macro"))

# =====================================================
# ==== Langkah 5 — Evaluasi Akhir di Test Set ====
# =====================================================
print("\n=== LANGKAH 5: EVALUASI AKHIR (TEST SET) ===")

final_model = best_rf  # bisa juga pipe_lr jika baseline lebih baik

y_test_pred = final_model.predict(X_test)
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# ROC-AUC dan visualisasi
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_test_proba)
    print("ROC-AUC(test):", auc)

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)
    plt.close()

# =====================================================
# ==== Langkah 6 — Simpan Model ====
# =====================================================
print("\n=== LANGKAH 6: SIMPAN MODEL ===")
joblib.dump(final_model, "model.pkl")
print("✅ Model tersimpan ke file: model.pkl")

print("\n=== SELESAI ===")
