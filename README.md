# BootcampAI 2026 — Face Anti-Spoofing Pipeline

Workspace proyek **Face Anti-Spoofing** multi-kelas (6 kategori: `realperson`, `fake_mannequin`, `fake_mask`, `fake_printed`, `fake_screen`, `fake_unknown`).

## Struktur Workspace

```text
BootcampAI2026/
├── Convnext/
│   ├── Master_Optuna_KFold_AntiSpoofing_v2_Documented_savedrive_freeze_backboneOutput.ipynb  # Pipeline utama ConvNeXt-Tiny 5-Fold CV + EDA + Grad-CAM
│   ├── Training_Pipeline_Butkem_AI_v4_Simplified.ipynb                                       # Pipeline 2-Stage EfficientNet-B0 (Train/Val/Test)
│   ├── artifacts/                                                                            # Artefak model & config.json (EfficientNet)
│   ├── saved_models/                                                                         # Checkpoint model 5-Fold ConvNeXt (di-ignore di Git)
│   └── train/                                                                                # Dataset hasil relabeling (6 folder kelas bersih)
├── data/
│   └── relabeling_csv/                                                                       # 6 CSV relabeling per kategori + sorting_report.csv
├── train/                                                                                    # Dataset mentah sebelum relabeling
├── test/                                                                                     # Dataset uji (404 gambar)
├── hasil_analisis/                                                                           # Output ekstraksi embedding & deteksi mislabel SigLIP
├── archive/                                                                                  # Arsip duplikat CSV lama & salinan notebook versi terdahulu
├── tests/
│   └── test_methods.py                                                                       # Unit & methodology verification tests
├── siglip_mislabel_detection.py                                                              # Deteksi kandidat mislabel dengan SigLIP + LOO Trimmed Centroid
├── dataset_sorter.py                                                                         # Tool otomatis sortir/relabeling dataset dari CSV
└── requirements.txt                                                                          # Daftar dependensi Python
```

## Ringkasan Perbaikan Metodologi

1. **`siglip_mislabel_detection.py`**
   - **Perbaikan Bug Cache Resume**: Menormalkan dimensi array embedding `(N, D)` dan path separator saat memuat `embedding_cache.npz` sehingga tidak terjadi `ValueError` saat melanjutkan ekstraksi batch.
   - **Trimmed Centroid & Deduplikasi MD5**: Menghapus duplikat MD5 di dalam kelas dan membuang 10% outlier terjauh saat mengestimasi centroid agar prototipe kelas tidak tertarik oleh sampel yang salah label.
   - **Leave-One-Out (LOO) Similarity**: Mengoreksi `own_class_similarity` menggunakan LOO centroid agar sampel tidak dibandingkan terhadap dirinya sendiri.

2. **`dataset_sorter.py`**
   - Dipindahkan keluar dari `Convnext/train/` agar folder dataset murni hanya berisi 6 subfolder kelas.
   - Melakukan *pre-scan* indeks penomoran di seluruh folder tujuan sebelum memindahkan file dan mendukung pemeriksaan duplikat MD5 (`--skip_exact_duplicates_in_target`).

3. **`Convnext/Master_Optuna_KFold_AntiSpoofing_v2_Documented_savedrive_freeze_backboneOutput.ipynb`**
   - **Perbaikan Rumus `FocalLoss`**: Menghitung probabilitas $p_t = \exp(-\text{CE}_{\text{unweighted}})$ sebelum dikalikan bobot kelas $\alpha_t$ (sebelumnya `weight=alpha` dimasukkan ke `CrossEntropyLoss` sehingga $p_t$ terdistorsi menjadi $(p_t)^{\alpha_t}$).
   - **Pencegahan Data Leakage pada Deduplikasi**: Membuang seluruh gambar yang memiliki MD5 identik namun berbeda label kelas (*cross-class conflicting duplicates*, 10 hash) dan hanya menyimpan 1 salinan untuk *within-class duplicates*.
   - **Per-Fold Class Weights**: Menghitung `class_weights` di dalam masing-masing fold hanya dari `train_df` fold tersebut.
   - **Kesesuaian Fold saat Evaluasi Akhir**: `evaluate_model` dan `plot_normalized_confusion_matrix` kini otomatis mengevaluasi `best_model_path` pada fold validasi yang sesuai (bukan selalu `break` di Fold 1).

4. **`Convnext/Training_Pipeline_Butkem_AI_v4_Simplified.ipynb`**
   - **Deduplikasi MD5 Sebelum Split**: Menghapus duplikat MD5 dan konflik lintas kelas sebelum `train_test_split` agar gambar identik tidak bocor antara `train_df`, `val_df`, dan `test_df`.
   - **Menghindari Double Imbalance Weighting**: Tidak menggandakan `WeightedRandomSampler` dengan `CrossEntropyLoss(weight=...)` secara bersamaan.
   - **Perbaikan Transisi Stage 1 → Stage 2 & `EarlyStopping`**: Menambahkan `early_stopper.reset()` (mereset `counter` dan `should_stop`) serta me-reload checkpoint terbaik dari Stage 1 sebelum melakukan *unfreeze* backbone di Stage 2.

## Cara Menjalankan

```bash
# 1. Jalankan verifikasi unit & metodologi
python tests/test_methods.py

# 2. Preview (dry-run) sortir dataset dari CSV
python dataset_sorter.py --base_dir train --csv_dir data/relabeling_csv --dry_run

# 3. Jalankan deteksi mislabel berbasis SigLIP
python siglip_mislabel_detection.py --dataset_root Convnext/train --output_dir hasil_analisis
```
