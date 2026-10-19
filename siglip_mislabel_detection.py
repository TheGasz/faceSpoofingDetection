"""
Script untuk membantu proses relabeling dataset anti spoofing wajah
menggunakan feature extraction dari SigLIP.

Alur kerja:
1. Load model SigLIP (image encoder saja)
2. Ekstraksi embedding untuk seluruh gambar di 6 folder kelas
3. Hitung centroid tiap kelas
4. Hitung cosine similarity tiap gambar ke seluruh centroid
5. Hitung margin (similarity ke kelas asal dikurangi similarity tertinggi ke kelas lain)
6. Urutkan kandidat mislabel berdasarkan margin paling mencurigakan
7. Visualisasi sebaran embedding dengan UMAP
8. Simpan hasil ke CSV untuk direview manual

Struktur folder yang diharapkan:
dataset_root/
    fake_mannequin/
        img1.jpg
        img2.jpg
    fake_mask/
        ...
    fake_printed/
        ...
    fake_screen/
        ...
    fake_unknown/
        ...
    realperson/
        ...
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# ============================================================
# KONFIGURASI
# ============================================================

DATASET_ROOT = "D:/Bagas/BootcampAI2026/train"  # ganti sesuai lokasi folder dataset kamu
MODEL_NAME = "google/siglip-so400m-patch14-384"  # bisa diganti versi lain sesuai kebutuhan
OUTPUT_DIR = "hasil_analisis"
CLASS_NAMES = [
    "fake_mannequin",
    "fake_mask",
    "fake_printed",
    "fake_screen",
    "fake_unknown",
    "realperson",
]

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print(f"Memuat model {MODEL_NAME} pada device {DEVICE}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return processor, model


# ============================================================
# KUMPULKAN PATH GAMBAR
# ============================================================

def collect_image_paths(dataset_root, class_names):
    records = []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for class_name in class_names:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"Peringatan: folder {class_dir} tidak ditemukan, dilewati")
            continue
        paths = [
            p for p in glob.glob(os.path.join(class_dir, "*"))
            if p.lower().endswith(valid_ext)
        ]
        for p in paths:
            records.append({"path": p, "label": class_name})
    df = pd.DataFrame(records)
    print(f"Total gambar ditemukan: {len(df)}")
    print(df["label"].value_counts())
    return df


# ============================================================
# EKSTRAKSI EMBEDDING
# ============================================================

def extract_embeddings(df, processor, model, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "embedding_cache.npz")

    paths = df["path"].tolist()
    labels = df["label"].tolist()
    total_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE

    # load cache kalau sudah ada dari proses sebelumnya
    done_paths = set()
    cached_embeddings = []
    cached_paths = []
    cached_labels = []

    if os.path.exists(cache_file):
        print(f"Ditemukan cache di {cache_file}, memuat progres sebelumnya", flush=True)
        cache_data = np.load(cache_file, allow_pickle=True)
        cached_embeddings = list(cache_data["embeddings"])
        cached_paths = list(cache_data["paths"])
        cached_labels = list(cache_data["labels"])
        done_paths = set(cached_paths)
        print(f"Sudah ada {len(done_paths)} gambar yang pernah diproses, akan dilewati", flush=True)

    all_embeddings = list(cached_embeddings)
    all_paths = list(cached_paths)
    all_labels = list(cached_labels)

    for batch_idx, i in enumerate(tqdm(range(0, len(paths), BATCH_SIZE), desc="Ekstraksi embedding")):
        batch_paths = paths[i:i + BATCH_SIZE]
        batch_labels = labels[i:i + BATCH_SIZE]

        # lewati gambar yang sudah pernah diproses di run sebelumnya
        batch_paths_filtered = []
        batch_labels_filtered = []
        for p, l in zip(batch_paths, batch_labels):
            if p not in done_paths:
                batch_paths_filtered.append(p)
                batch_labels_filtered.append(l)

        if not batch_paths_filtered:
            continue

        print(f"[batch {batch_idx + 1}/{total_batches}] membuka {len(batch_paths_filtered)} gambar", flush=True)

        images = []
        valid_paths = []
        valid_labels = []
        for p, l in zip(batch_paths_filtered, batch_labels_filtered):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
                valid_labels.append(l)
            except Exception as e:
                print(f"Gagal membuka {p}: {e}", flush=True)

        if not images:
            continue

        print(f"[batch {batch_idx + 1}/{total_batches}] preprocessing gambar", flush=True)
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)

        print(f"[batch {batch_idx + 1}/{total_batches}] forward pass ke model", flush=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

            # beberapa versi transformers mengembalikan objek output lengkap
            # (misalnya BaseModelOutputWithPooling), bukan tensor langsung
            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                elif hasattr(image_features, "last_hidden_state"):
                    image_features = image_features.last_hidden_state.mean(dim=1)
                else:
                    raise TypeError(
                        f"Tipe output tidak dikenali dari get_image_features: {type(image_features)}"
                    )

            image_features = torch.nn.functional.normalize(image_features, dim=-1)

        print(f"[batch {batch_idx + 1}/{total_batches}] selesai, embedding shape {tuple(image_features.shape)}", flush=True)

        batch_embeddings = image_features.cpu().numpy()
        all_embeddings.append(batch_embeddings)
        all_paths.extend(valid_paths)
        all_labels.extend(valid_labels)

        # simpan cache setiap selesai satu batch, supaya kalau proses berhenti
        # di tengah jalan, tidak perlu mengulang dari awal
        np.savez(
            cache_file,
            embeddings=np.concatenate(all_embeddings, axis=0),
            paths=np.array(all_paths, dtype=object),
            labels=np.array(all_labels, dtype=object),
        )

    embeddings = np.concatenate(all_embeddings, axis=0)
    result_df = pd.DataFrame({"path": all_paths, "label": all_labels})
    return result_df, embeddings


# ============================================================
# HITUNG CENTROID TIAP KELAS
# ============================================================

def compute_centroids(embeddings, labels, class_names):
    centroids = {}
    for class_name in class_names:
        idx = [i for i, l in enumerate(labels) if l == class_name]
        if not idx:
            continue
        class_embeddings = embeddings[idx]
        centroid = class_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[class_name] = centroid
    return centroids


# ============================================================
# HITUNG SIMILARITY DAN MARGIN
# ============================================================

def compute_similarity_and_margin(df, embeddings, centroids, class_names):
    centroid_matrix = np.stack([centroids[c] for c in class_names])  # shape (num_class, dim)
    similarity_matrix = embeddings @ centroid_matrix.T  # cosine similarity karena sudah dinormalisasi

    sim_df = pd.DataFrame(similarity_matrix, columns=[f"sim_{c}" for c in class_names])
    result = pd.concat([df.reset_index(drop=True), sim_df], axis=1)

    own_sim = []
    best_other_sim = []
    best_other_label = []

    for i, row in result.iterrows():
        label = row["label"]
        sims = {c: row[f"sim_{c}"] for c in class_names}
        current_sim = sims[label]
        other_sims = {c: v for c, v in sims.items() if c != label}
        best_c = max(other_sims, key=other_sims.get)
        best_v = other_sims[best_c]

        own_sim.append(current_sim)
        best_other_sim.append(best_v)
        best_other_label.append(best_c)

    result["own_class_similarity"] = own_sim
    result["best_other_class_similarity"] = best_other_sim
    result["best_other_class"] = best_other_label
    result["margin"] = result["own_class_similarity"] - result["best_other_class_similarity"]

    # margin negatif berarti gambar lebih mirip ke kelas lain daripada kelas asalnya sendiri
    # semakin negatif semakin mencurigakan
    result = result.sort_values("margin", ascending=True).reset_index(drop=True)
    return result


# ============================================================
# VISUALISASI UMAP
# ============================================================

def visualize_umap(embeddings, labels, output_path):
    try:
        import umap
    except ImportError:
        print("Package umap-learn belum terinstall, jalankan: pip install umap-learn --break-system-packages")
        return

    import matplotlib.pyplot as plt

    print("Menjalankan reduksi dimensi UMAP")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], label=lbl, s=8, alpha=0.6)

    plt.legend(markerscale=2)
    plt.title("Sebaran embedding SigLIP per kelas (UMAP)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Visualisasi UMAP disimpan di {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    processor, model = load_model()

    df = collect_image_paths(DATASET_ROOT, CLASS_NAMES)
    if len(df) == 0:
        print("Tidak ada gambar ditemukan, periksa kembali DATASET_ROOT")
        return

    df, embeddings = extract_embeddings(df, processor, model, cache_dir=OUTPUT_DIR)

    embeddings_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Embedding disimpan di {embeddings_path}")

    centroids = compute_centroids(embeddings, df["label"].tolist(), CLASS_NAMES)

    result = compute_similarity_and_margin(df, embeddings, centroids, CLASS_NAMES)

    csv_path = os.path.join(OUTPUT_DIR, "hasil_deteksi_mislabel.csv")
    result.to_csv(csv_path, index=False)
    print(f"Hasil lengkap disimpan di {csv_path}")

    # ambang batas margin bisa disesuaikan, ini contoh 0 artinya
    # similarity ke kelas lain sama atau lebih tinggi dari kelas asalnya sendiri
    suspicious = result[result["margin"] < 0]
    suspicious_path = os.path.join(OUTPUT_DIR, "kandidat_mislabel.csv")
    suspicious.to_csv(suspicious_path, index=False)
    print(f"Ditemukan {len(suspicious)} kandidat mislabel, disimpan di {suspicious_path}")

    umap_path = os.path.join(OUTPUT_DIR, "umap_visualization.png")
    visualize_umap(embeddings, df["label"].tolist(), umap_path)

    print("Selesai. Silakan review manual file kandidat_mislabel.csv sebelum melakukan relabel final.")


if __name__ == "__main__":
    main()