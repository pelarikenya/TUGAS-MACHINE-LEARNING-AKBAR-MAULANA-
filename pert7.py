# ============================================================
# 📘 Pertemuan 7 — Artificial Neural Network (ANN)
# ============================================================

# Langkah 1 — Siapkan Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Membaca data hasil Pertemuan 4
df = pd.read_csv("processed_kelulusan.csv")

# Pisahkan fitur dan target
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Standarisasi fitur numerik
sc = StandardScaler()
Xs = sc.fit_transform(X)

# Split data: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("📊 Data Shapes:")
print("Train :", X_train.shape)
print("Val   :", X_val.shape)
print("Test  :", X_test.shape)

# ============================================================
# Langkah 2 — Bangun Model ANN
# ============================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu", name="Hidden_1"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu", name="Hidden_2"),
    layers.Dense(1, activation="sigmoid", name="Output")  # klasifikasi biner
])

# Kompilasi model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="AUC")]
)

# Ringkasan arsitektur
model.summary()

# ============================================================
# Langkah 3 — Training dengan Early Stopping
# ============================================================
es = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# ============================================================
# Langkah 4 — Evaluasi di Test Set
# ============================================================
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Evaluasi di Test Set:")
print(f"Test Accuracy : {acc:.3f}")
print(f"Test AUC       : {auc:.3f}")

# Prediksi probabilitas dan kelas
y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

# Hitung metrik tambahan
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

print(f"F1-score (test): {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC (test): {roc_auc_score(y_test, y_proba):.3f}")

# ============================================================
# Langkah 5 — Visualisasi Learning Curve
# ============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve — ANN")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()

# ============================================================
# Langkah 6 — Simpan Model
# ============================================================
model.save("ann_kelulusan_model.h5")
print("\n💾 Model disimpan sebagai ann_kelulusan_model.h5")

# ============================================================
# Langkah 7 — Cek Inference (Prediksi Baru)
# ============================================================
import numpy as np
mdl = keras.models.load_model("ann_kelulusan_model.h5")

# Contoh data baru (fiktif)
sample = pd.DataFrame([{
    "IPK": 3.5,
    "Jumlah_Absensi": 3,
    "Waktu_Belajar_Jam": 8,
    "Rasio_Absensi": 3/14,
    "IPK_x_Study": 3.5*8
}])

# Skala fitur sebelum prediksi
sample_scaled = sc.transform(sample)
pred = mdl.predict(sample_scaled)
print("\n🔮 Prediksi baru (1=Lulus, 0=Tidak Lulus):", int(pred[0] >= 0.5))
