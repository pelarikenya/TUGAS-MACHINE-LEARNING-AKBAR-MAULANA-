pertemuan 5 
=== LANGKAH 1: MUAT DATA === Bentuk data: (14, 5) (3, 5) (3, 5)

=== LANGKAH 2: BASELINE MODEL === Baseline (LogReg) F1(val): 1.0 precision recall f1-score support

       0      1.000     1.000     1.000         1
       1      1.000     1.000     1.000         2

accuracy                          1.000         3
macro avg 1.000 1.000 1.000 3 weighted avg 1.000 1.000 1.000 3

=== LANGKAH 3: RANDOM FOREST === RandomForest F1(val): 1.0 precision recall f1-score support

       0      1.000     1.000     1.000         1
       1      1.000     1.000     1.000         2

accuracy                          1.000         3
macro avg 1.000 1.000 1.000 3 weighted avg 1.000 1.000 1.000 3

=== LANGKAH 4: GRID SEARCH TUNING === Fitting 5 folds for each of 12 candidates, totalling 60 fits Best params: {'clf__max_depth': None, 'clf__min_samples_split': 2} Best CV F1: 1.0 Best RF F1(val): 1.0

=== LANGKAH 5: EVALUASI AKHIR (TEST SET) === F1(test): 1.0 precision recall f1-score support

       0      1.000     1.000     1.000         2
       1      1.000     1.000     1.000         1

accuracy                          1.000         3
macro avg 1.000 1.000 1.000 3 weighted avg 1.000 1.000 1.000 3

Confusion matrix (test): [[2 0] [0 1]] ROC-AUC(test): 1.0

=== LANGKAH 6: SIMPAN MODEL === ✅ Model tersimpan ke file: model.pkl

=== SELESAI ===
