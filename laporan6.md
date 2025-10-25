pertemuan 6
=== LANGKAH 0/1: MEMUAT DATA === Sumber: processed_kelulusan.csv (pilihan A) Melakukan stratified split 70/15/15 dari processed_kelulusan.csv ... Shapes (split): (14, 5) (3, 5) (3, 5) Label unik (train): [0, 1]

=== LANGKAH 2: PIPELINE & BASELINE RANDOM FOREST === Kolom numerik: ['IPK', 'Jumlah_Absensi', 'Waktu_Belajar_Jam', 'Rasio_Absensi', 'IPK_x_Study'] Baseline RF — F1(val, macro): 1.0 Baseline classification report (val): precision recall f1-score support

       0      1.000     1.000     1.000         1
       1      1.000     1.000     1.000         2

accuracy                          1.000         3
macro avg 1.000 1.000 1.000 3 weighted avg 1.000 1.000 1.000 3

=== LANGKAH 3: VALIDASI SILANG (StratifiedKFold) === CV F1-macro (train): 1.0000 ± 0.0000

=== LANGKAH 4: GRID SEARCH TUNING RINGKAS === Fitting 5 folds for each of 12 candidates, totalling 60 fits Best params: {'clf__max_depth': None, 'clf__min_samples_split': 2} Best CV F1 (train): 1.0 Best RF — F1(val, macro): 1.0 Classification report (best, val): precision recall f1-score support

       0      1.000     1.000     1.000         1
       1      1.000     1.000     1.000         2

accuracy                          1.000         3
macro avg 1.000 1.000 1.000 3 weighted avg 1.000 1.000 1.000 3

=== LANGKAH 5: EVALUASI AKHIR (TEST SET) === F1(test, macro): 1.0 Classification report (test): precision recall f1-score support

       0      1.000     1.000     1.000         2
       1      1.000     1.000     1.000         1

accuracy                          1.000         3
macro avg 1.000 1.000 1.000 3 weighted avg 1.000 1.000 1.000 3

Confusion matrix (test): [[2 0] [0 1]] ROC-AUC(test): 1.0

=== LANGKAH 6: FEATURE IMPORTANCE === Top feature importances: num__Rasio_Absensi: 0.2267 num__IPK: 0.2100 num__Jumlah_Absensi: 0.1967 num__IPK_x_Study: 0.1933 num__Waktu_Belajar_Jam: 0.1733 Feature importance disimpan di outputs/feature_importance.csv

=== LANGKAH 7: SIMPAN MODEL === Model disimpan sebagai rf_model.pkl

=== LANGKAH 8: CEK INFERENCE LOKAL (CONTOH) === Contoh prediksi pada sample (median fitur): 1 proba: 0.5533333333333333

=== RINGKASAN AKHIR ===

Baseline RF + tuning (GridSearchCV) telah dilakukan.
Best CV F1 (train): 1.0000
F1 (test, macro): 1.0000
Files output: outputs/roc_test.png, outputs/pr_test.png, outputs/feature_importance.csv, rf_model.pkl
Selesai.
