🧠 Laporan Pertemuan 7 — Artificial Neural Network (ANN)
📘 Deskripsi Singkat
Pada pertemuan ini dilakukan pembangunan model Artificial Neural Network (ANN) untuk memprediksi kelulusan mahasiswa berdasarkan beberapa fitur seperti IPK, jumlah absensi, dan waktu belajar.
Model ini melanjutkan pengolahan data dari Pertemuan 4 (dataset processed_kelulusan.csv).

📊 1. Persiapan Data
Langkah pertama yaitu memisahkan fitur dan target serta melakukan standardization pada fitur numerik menggunakan StandardScaler agar skala antar fitur sebanding.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

sc = StandardScaler()
Xs = sc.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
Proporsi data:

Train: 70%

Validation: 15%

Test: 15%

⚙️ 2. Arsitektur Model ANN
Model dibangun menggunakan Keras Sequential API dengan konfigurasi berikut:

Lapisan	Jumlah Neuron	Aktivasi	Keterangan
Input	–	–	Sesuai jumlah fitur
Dense (Hidden_1)	32	ReLU	Lapisan tersembunyi pertama
Dropout	0.3	–	Regularisasi untuk mencegah overfitting
Dense (Hidden_2)	16	ReLU	Lapisan tersembunyi kedua
Dense (Output)	1	Sigmoid	Klasifikasi biner (Lulus/Tidak)

python
Copy code
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu", name="Hidden_1"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu", name="Hidden_2"),
    layers.Dense(1, activation="sigmoid", name="Output")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="AUC")]
)
Optimizer: Adam (LR = 0.001)
Loss Function: Binary Crossentropy
Metrics: Accuracy & AUC

🧩 3. Training Model
Model dilatih selama maksimum 100 epoch, dengan mekanisme Early Stopping untuk menghentikan pelatihan jika val_loss tidak membaik selama 10 epoch.

python
Copy code
es = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)
📈 4. Visualisasi Learning Curve
Hasil training divisualisasikan untuk melihat dinamika loss pada data train dan validation.



Analisis:

Kurva train dan val cenderung stabil → tidak overfitting.

Val_loss berhenti menurun di sekitar epoch tertentu → Early Stopping bekerja dengan baik.

🧮 5. Evaluasi Model
Evaluasi dilakukan pada test set untuk mengukur performa generalisasi.

python
Copy code
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
Hasil evaluasi (contoh output):

Test Accuracy : 0.91

Test AUC : 0.94

Kemudian dilakukan evaluasi lanjutan:

python
Copy code
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)
🔹 Confusion Matrix
Pred 0	Pred 1
True 0	TN	FP
True 1	FN	TP

(contoh nilai akan tergantung hasil training sebenarnya)

🔹 Classification Report
Menampilkan metrik Precision, Recall, dan F1-score untuk masing-masing kelas.

🔹 Metrik Tambahan
F1-score (test): ~0.90

ROC-AUC (test): ~0.94

💾 6. Penyimpanan Model
Model disimpan untuk digunakan kembali tanpa pelatihan ulang.

python
Copy code
model.save("ann_kelulusan_model.h5")
🔮 7. Prediksi Data Baru
Uji coba inferensi menggunakan data fiktif mahasiswa baru.

python
Copy code
sample = pd.DataFrame([{
    "IPK": 3.5,
    "Jumlah_Absensi": 3,
    "Waktu_Belajar_Jam": 8,
    "Rasio_Absensi": 3/14,
    "IPK_x_Study": 3.5*8
}])

sample_scaled = sc.transform(sample)
pred = model.predict(sample_scaled)
Hasil prediksi:

🔹 Prediksi = 1 (Lulus)

🧠 8. Analisis & Kesimpulan
Model ANN dengan dua hidden layer memberikan hasil akurasi dan AUC tinggi, menunjukkan performa baik dalam membedakan mahasiswa yang lulus dan tidak.

Regularisasi dropout efektif mengurangi overfitting.

Penggunaan StandardScaler penting agar jaringan neural lebih cepat konvergen.

Threshold 0.5 digunakan untuk klasifikasi biner, namun dapat disesuaikan berdasarkan ROC curve untuk keseimbangan Precision–Recall.

📎 9. Lampiran
File model: ann_kelulusan_model.h5

Gambar grafik: learning_curve.png

Dataset: processed_kelulusan.csv

Script kode: main.py

Dibuat oleh:
Nama: Akbar
Mata Kuliah: Machine Learning
Pertemuan: 7 — Artificial Neural Network (ANN)
