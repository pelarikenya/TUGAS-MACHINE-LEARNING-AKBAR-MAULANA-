# ==========================================================
# PERTEMUAN 4 — DATA COLLECTION, CLEANING, EDA, FEATURE ENGINEERING
# ==========================================================

# === LANGKAH 1: IMPORT LIBRARY UTAMA ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === LANGKAH 2: COLLECTION (Membaca Dataset) ===
print("=== LANGKAH 2: COLLECTION ===")
df = pd.read_csv("kelulusan_mahasiswa.csv")

# Info dan data awal
print(df.info())
print("\nData Awal:")
print(df.head())

# === LANGKAH 3: CLEANING ===
print("\n=== LANGKAH 3: CLEANING ===")
# Cek missing value
print("Missing Values:\n", df.isnull().sum())

# Hapus duplikat jika ada
df = df.drop_duplicates()
print("\nSetelah drop_duplicates, jumlah data:", len(df))

# Visualisasi boxplot untuk deteksi outlier
plt.figure(figsize=(5, 3))
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.tight_layout()
plt.savefig("boxplot_ipk.png", dpi=120)
plt.close()

# === LANGKAH 4: EXPLORATORY DATA ANALYSIS (EDA) ===
print("\n=== LANGKAH 4: EDA ===")
# Statistik deskriptif
print(df.describe())

# Histogram distribusi IPK
plt.figure(figsize=(5, 3))
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.xlabel("IPK")
plt.tight_layout()
plt.savefig("hist_ipk.png", dpi=120)
plt.close()

# Scatterplot IPK vs Waktu Belajar
plt.figure(figsize=(5, 3))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar (berdasarkan Kelulusan)")
plt.tight_layout()
plt.savefig("scatter_ipk_study.png", dpi=120)
plt.close()

# Heatmap korelasi
plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi")
plt.tight_layout()
plt.savefig("heatmap_korelasi.png", dpi=120)
plt.close()

# === LANGKAH 5: FEATURE ENGINEERING ===
print("\n=== LANGKAH 5: FEATURE ENGINEERING ===")
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan hasil cleaning dan feature engineering
df.to_csv("processed_kelulusan.csv", index=False)
print("File processed_kelulusan.csv berhasil disimpan!")

# === LANGKAH 6: SPLITTING DATASET ===
print("\n=== LANGKAH 6: SPLITTING DATASET ===")

X = df.drop('Lulus', axis=1)
y = df['Lulus']

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train set :", X_train.shape)
print("Validasi  :", X_val.shape)
print("Test set  :", X_test.shape)

print("\n=== PROSES SELESAI ===")
print("📁 File processed_kelulusan.csv sudah berisi data bersih & fitur baru.")
print("📊 Grafik disimpan sebagai: boxplot_ipk.png, hist_ipk.png, scatter_ipk_study.png, heatmap_korelasi.png")
print("✅ Dataset terbagi menjadi Train, Validation, dan Test.")
