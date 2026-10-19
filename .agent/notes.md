# Project Notes & Conventions — BootcampAI2026 (Face Anti-Spoofing)

## 1. Gambaran Umum Proyek
Proyek klasifikasi **Face Anti-Spoofing** 6 kelas:
- `realperson` (Real)
- `fake_mannequin`, `fake_mask`, `fake_printed`, `fake_screen`, `fake_unknown` (Spoof)

## 2. Struktur Dataset
- `train/`: Dataset mentah sebelum relabeling (1.652 gambar).
- `Convnext/train/`: Dataset setelah proses relabeling menggunakan `dataset_sorter.py` (1.652 gambar total; terdapat 339 baris duplikat MD5 dari 161 hash unik, termasuk 10 hash konflik lintas kelas).
- `test/`: Dataset uji tanpa label (404 gambar).

## 3. Temuan Audit Metodologi & Bug
1. **`siglip_mislabel_detection.py`**:
   - **Bug Resume Cache (`extract_embeddings`)**: `list(cache_data["embeddings"])` mengubah array 2D `(N, D)` menjadi list array 1D `(D,)`, sedangkan batch baru menambahkan array 2D `(B, D)` sehingga `np.concatenate(..., axis=0)` crash (`ValueError`).
   - **Bias Centroid**: Perhitungan centroid kelas memasukkan sampel itu sendiri (*self-inclusion bias*) dan sensitif terhadap outlier/duplikat (~14% mislabel). Diperbaiki dengan deduplikasi hash, *iterative trimmed centroid*, dan *Leave-One-Out (LOO) centroid* untuk similarity ke kelas asal.
   - **Performa**: Vektorisasi perhitungan cosine similarity dan margin tanpa `iterrows()`.
2. **`dataset_sorter.py`**:
   - Sebelumnya diletakkan di dalam `Convnext/train/` bersama CSV dan `__pycache__/`, yang mengotori direktori dataset `ImageFolder`.
   - Ditambahkan validasi hash MD5 agar tidak menimpa/membuat duplikat buta saat pemindahan file dan mendukung path konfigurasi yang bersih.
3. **`Convnext/Master_Optuna_KFold_AntiSpoofing_v2_Documented_savedrive_freeze_backboneOutput.ipynb`**:
   - **Bug Matematis `FocalLoss`**: Menggunakan `nn.CrossEntropyLoss(weight=alpha)` untuk menghitung `log_pt = -self.ce(inputs, targets)`, sehingga `pt = torch.exp(log_pt)` menjadi $(p_t)^{\alpha_t}$ bukan $p_t$. Diperbaiki menjadi `ce_loss = F.cross_entropy(inputs, targets, reduction='none')`, `pt = torch.exp(-ce_loss)`, lalu dikalikan `alpha_t` di luar.
   - **Data Leakage `class_weights`**: Sebelumnya dihitung di seluruh `df` sebelum split K-Fold. Diperbaiki agar dihitung di dalam tiap fold hanya dari `train_df`.
   - **Cross-Class Duplicate Leakage**: `drop_duplicates(subset=['hash'], keep='first')` sebelumnya menyimpan salah satu label secara arbitrer untuk 10 gambar yang identik di 2 kelas berbeda. Diperbaiki dengan membuang hash yang memiliki konflik label lintas kelas (*cross-class duplicates*) dan hanya melakukan `keep='first'` untuk *within-class duplicates*.
   - **Bug Evaluasi Fold (`evaluate_model` & `plot_normalized_confusion_matrix`)**: Sebelumnya selalu mengevaluasi `best_model_path` pada Fold 1 (`break` pada iterasi pertama `skf.split`), yang bocor ke data training jika `best_fold_idx != 1`. Diperbaiki agar mengevaluasi tepat pada `val_idx` dari `best_fold_idx` serta mendukung evaluasi Out-Of-Fold (OOF) gabungan 5 fold.
4. **`Convnext/Training_Pipeline_Butkem_AI_v4_Simplified.ipynb`**:
   - **Severe Data Leakage pada Split**: Tidak melakukan deduplikasi MD5 sebelum `train_test_split`, sehingga 339 gambar duplikat bocor antara train, val, dan test set. Diperbaiki dengan deduplikasi MD5 (termasuk penghapusan cross-class conflict) sebelum split.
   - **Double Imbalance Compensation**: Memakai `WeightedRandomSampler` sekaligus `CrossEntropyLoss(weight=class_weight_tensor)` (mengkuadratkan bobot kelas minoritas). Diperbaiki dengan memilih satu mekanisme yang konsisten (tidak menggandakan bobot).
   - **Bug EarlyStopping & Transisi Stage 1 -> Stage 2**: `early_stopper.should_stop` tidak di-reset ke `False` dan bobot terbaik Stage 1 tidak di-load sebelum memulai Stage 2. Diperbaiki dengan method `early_stopper.reset_counter()` dan reload checkpoint terbaik Stage 1.
   - **Typo `PYTHONSEED` & Deprecated AMP API**: Diperbaiki menjadi `PYTHONHASHSEED` dan `torch.amp.autocast('cuda')` / `torch.amp.GradScaler('cuda')`.
   - **Visualisasi Grad-CAM**: Diperbaiki agar menampilkan label asli vs prediksi + confidence serta `ylabel` tidak hilang akibat `axis('off')`.
