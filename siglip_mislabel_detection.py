"""
Script untuk membantu proses relabeling dataset anti spoofing wajah
menggunakan feature extraction dari SigLIP.

Alur kerja:
1. Load model SigLIP (image encoder saja)
2. Ekstraksi embedding untuk seluruh gambar di 6 folder kelas (dengan resume cache yang aman)
3. Hitung centroid tiap kelas secara robust (deduplikasi MD5 + iterative trimmed centroid)
4. Hitung cosine similarity tiap gambar ke seluruh centroid (dengan koreksi Leave-One-Out untuk kelas asal)
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

import argparse
import glob
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# ============================================================
# KONFIGURASI DEFAULT
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.join(SCRIPT_DIR, "train"))
DEFAULT_MODEL_NAME = os.environ.get("SIGLIP_MODEL_NAME", "google/siglip-so400m-patch14-384")
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(SCRIPT_DIR, "hasil_analisis"))

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


def _norm_path(path: str) -> str:
    """Normalisasi path agar konsisten lintas separator ('/' vs '\\')."""
    return os.path.normcase(os.path.normpath(str(path)))


def _compute_md5(filepath: str) -> str:
    """Hitung MD5 hash dari file gambar untuk deteksi duplikat."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_name: str = DEFAULT_MODEL_NAME, device: str = DEVICE):
    """Memuat processor dan image model SigLIP."""
    print(f"Memuat model {model_name} pada device {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


# ============================================================
# KUMPULKAN PATH GAMBAR
# ============================================================

def collect_image_paths(dataset_root: str, class_names: List[str]) -> pd.DataFrame:
    """Mengumpulkan seluruh file gambar valid dari setiap subfolder kelas."""
    records = []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for class_name in class_names:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"Peringatan: folder {class_dir} tidak ditemukan, dilewati")
            continue
        paths = sorted([
            p for p in glob.glob(os.path.join(class_dir, "*"))
            if os.path.isfile(p) and p.lower().endswith(valid_ext)
        ])
        for p in paths:
            records.append({
                "path": p,
                "label": class_name,
                "md5": _compute_md5(p),
            })
    df = pd.DataFrame(records)
    print(f"Total gambar ditemukan: {len(df)}")
    if len(df) > 0:
        print(df["label"].value_counts())
    return df


# ============================================================
# EKSTRAKSI EMBEDDING
# ============================================================

def extract_embeddings(
    df: pd.DataFrame,
    processor,
    model,
    cache_dir: str,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Mengekstrak L2-normalized image embedding menggunakan SigLIP dengan dukungan
    resume cache yang aman terhadap dimensi array 2D dan perbedaan path separator.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "embedding_cache.npz")

    paths = df["path"].tolist()
    labels = df["label"].tolist()
    total_batches = (len(paths) + batch_size - 1) // batch_size

    # Map normalized_path -> (embedding_1d, original_path, label)
    cache_map: Dict[str, Tuple[np.ndarray, str, str]] = {}

    if os.path.exists(cache_file):
        print(f"Ditemukan cache di {cache_file}, memuat progres sebelumnya", flush=True)
        cache_data = np.load(cache_file, allow_pickle=True)
        raw_embs = np.asarray(cache_data["embeddings"], dtype=np.float32)
        raw_paths = cache_data["paths"].tolist()
        raw_labels = cache_data["labels"].tolist()

        if raw_embs.ndim == 2 and len(raw_paths) == raw_embs.shape[0]:
            for emb, p, l in zip(raw_embs, raw_paths, raw_labels):
                cache_map[_norm_path(p)] = (emb, str(p), str(l))
        print(f"Sudah ada {len(cache_map)} gambar unik di cache, akan dilewati", flush=True)

    for batch_idx, i in enumerate(tqdm(range(0, len(paths), batch_size), desc="Ekstraksi embedding")):
        batch_paths = paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        batch_paths_filtered = []
        batch_labels_filtered = []
        for p, l in zip(batch_paths, batch_labels):
            if _norm_path(p) not in cache_map:
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
                with Image.open(p) as raw_img:
                    img = raw_img.convert("RGB")
                images.append(img)
                valid_paths.append(p)
                valid_labels.append(l)
            except Exception as e:
                print(f"Gagal membuka {p}: {e}", flush=True)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                    image_features = image_features.pooler_output
                elif hasattr(image_features, "last_hidden_state"):
                    image_features = image_features.last_hidden_state.mean(dim=1)
                else:
                    raise TypeError(
                        f"Tipe output tidak dikenali dari get_image_features: {type(image_features)}"
                    )

            image_features = torch.nn.functional.normalize(image_features, dim=-1)

        batch_embeddings = image_features.cpu().numpy().astype(np.float32)
        for emb, p, l in zip(batch_embeddings, valid_paths, valid_labels):
            cache_map[_norm_path(p)] = (emb, p, l)

        # Simpan seluruh isi cache_map sebagai matriks 2D (N, D)
        saved_embs = np.stack([v[0] for v in cache_map.values()], axis=0)
        saved_paths = np.array([v[1] for v in cache_map.values()], dtype=object)
        saved_labels = np.array([v[2] for v in cache_map.values()], dtype=object)
        np.savez(
            cache_file,
            embeddings=saved_embs,
            paths=saved_paths,
            labels=saved_labels,
        )

    # Susun kembali sesuai urutan df yang berhasil diekstrak
    ordered_rows = []
    ordered_embs = []
    for _, row in df.iterrows():
        key = _norm_path(row["path"])
        if key in cache_map:
            emb, _, _ = cache_map[key]
            ordered_embs.append(emb)
            ordered_rows.append(row.to_dict())

    if not ordered_embs:
        raise RuntimeError("Tidak ada embedding yang berhasil diekstrak.")

    embeddings = np.stack(ordered_embs, axis=0)
    # Pastikan setiap baris ter-normalisasi L2
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    result_df = pd.DataFrame(ordered_rows).reset_index(drop=True)
    return result_df, embeddings


# ============================================================
# HITUNG CENTROID TIAP KELAS (ROBUST TRIMMED PROTOTYPE)
# ============================================================

def compute_centroids(
    embeddings: np.ndarray,
    labels: List[str],
    class_names: List[str],
    md5_hashes: Optional[List[str]] = None,
    trim_ratio: float = 0.10,
) -> Dict[str, np.ndarray]:
    """
    Menghitung centroid tiap kelas secara robust:
    1. Jika md5_hashes diberikan, duplikat persis di dalam kelas yang sama hanya dihitung
       satu kali agar tidak mendistorsi arah centroid.
    2. Jika trim_ratio > 0, dilakukan 1 tahap iterative trimming: buang `trim_ratio`
       sampel dengan similarity terendah terhadap mean awal (mengurangi kontaminasi
       sampel mislabel saat mengestimasi pusat kelas).
    """
    labels_arr = np.asarray(labels)
    centroids: Dict[str, np.ndarray] = {}

    for class_name in class_names:
        idx = np.where(labels_arr == class_name)[0]
        if len(idx) == 0:
            continue

        # Deduplikasi berdasarkan hash MD5 di dalam kelas agar duplikat tidak memberi bobot ganda
        if md5_hashes is not None:
            seen_hashes = set()
            unique_idx = []
            for i in idx:
                h = md5_hashes[i]
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_idx.append(i)
            idx = np.asarray(unique_idx, dtype=int)

        class_embeddings = embeddings[idx]
        init_centroid = class_embeddings.mean(axis=0)
        init_centroid = init_centroid / np.clip(np.linalg.norm(init_centroid), 1e-12, None)

        if 0.0 < trim_ratio < 0.5 and len(class_embeddings) >= 10:
            sims = class_embeddings @ init_centroid
            keep_n = max(1, int(round(len(class_embeddings) * (1.0 - trim_ratio))))
            keep_indices = np.argsort(sims)[-keep_n:]
            refined = class_embeddings[keep_indices].mean(axis=0)
            centroid = refined / np.clip(np.linalg.norm(refined), 1e-12, None)
        else:
            centroid = init_centroid

        centroids[class_name] = centroid

    return centroids


# ============================================================
# HITUNG SIMILARITY DAN MARGIN (VEKTORISASI + LOO CORRECTION)
# ============================================================

def compute_similarity_and_margin(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    centroids: Dict[str, np.ndarray],
    class_names: List[str],
    use_loo_own_similarity: bool = True,
) -> pd.DataFrame:
    """
    Menghitung cosine similarity tiap gambar ke seluruh centroid kelas dan marginnya.
    Jika `use_loo_own_similarity=True`, similarity ke kelas asal dikoreksi dengan
    Leave-One-Out (LOO) supaya sampel ke-i tidak dibandingkan terhadap dirinya sendiri.
    """
    active_classes = [c for c in class_names if c in centroids]
    if len(active_classes) < 2:
        raise ValueError("Minimal 2 kelas harus memiliki centroid untuk menghitung margin.")

    centroid_matrix = np.stack([centroids[c] for c in active_classes], axis=0)  # (C, D)
    similarity_matrix = embeddings @ centroid_matrix.T  # (N, C)

    class_to_col = {c: idx for idx, c in enumerate(active_classes)}
    labels_list = df["label"].tolist()
    n_samples = len(df)

    # Koreksi Leave-One-Out (LOO) pada similarity kelas asal
    own_sim = np.zeros(n_samples, dtype=np.float32)
    labels_arr = np.asarray(labels_list)

    if use_loo_own_similarity:
        # Hitung sum embedding per kelas untuk mempercepat LOO mean: (sum_c - e_i) / (N_c - 1)
        class_sums = {}
        class_counts = {}
        for c in active_classes:
            c_mask = (labels_arr == c)
            class_sums[c] = embeddings[c_mask].sum(axis=0)
            class_counts[c] = int(c_mask.sum())

        for i, lbl in enumerate(labels_list):
            col_idx = class_to_col[lbl]
            if class_counts[lbl] > 1:
                loo_vec = class_sums[lbl] - embeddings[i]
                loo_norm = np.linalg.norm(loo_vec)
                if loo_norm > 1e-12:
                    loo_centroid = loo_vec / loo_norm
                    own_sim[i] = float(np.dot(embeddings[i], loo_centroid))
                    similarity_matrix[i, col_idx] = own_sim[i]
                else:
                    own_sim[i] = float(similarity_matrix[i, col_idx])
            else:
                own_sim[i] = float(similarity_matrix[i, col_idx])
    else:
        for i, lbl in enumerate(labels_list):
            own_sim[i] = float(similarity_matrix[i, class_to_col[lbl]])

    sim_df = pd.DataFrame(similarity_matrix, columns=[f"sim_{c}" for c in active_classes])
    result = pd.concat([df.reset_index(drop=True), sim_df], axis=1)

    # Cari similarity tertinggi di luar kelas asal secara vektorisasi
    masked_sim = similarity_matrix.copy()
    for i, lbl in enumerate(labels_list):
        masked_sim[i, class_to_col[lbl]] = -np.inf

    best_other_idx = np.argmax(masked_sim, axis=1)
    best_other_sim = masked_sim[np.arange(n_samples), best_other_idx]
    best_other_label = [active_classes[idx] for idx in best_other_idx]

    result["own_class_similarity"] = own_sim
    result["best_other_class_similarity"] = best_other_sim
    result["best_other_class"] = best_other_label
    result["margin"] = result["own_class_similarity"] - result["best_other_class_similarity"]

    # Tandai apakah file ini merupakan duplikat MD5 (terutama duplikat lintas kelas)
    if "md5" in result.columns:
        dup_counts = result.groupby("md5")["label"].transform("nunique")
        result["is_cross_class_duplicate"] = dup_counts > 1
        result["is_exact_duplicate"] = result.duplicated(subset=["md5"], keep=False)

    # Margin negatif berarti gambar lebih mirip ke kelas lain daripada kelas asalnya sendiri
    result = result.sort_values("margin", ascending=True).reset_index(drop=True)
    return result


# ============================================================
# VISUALISASI UMAP
# ============================================================

def visualize_umap(embeddings: np.ndarray, labels: List[str], output_path: str):
    try:
        import umap
    except ImportError:
        print("Package umap-learn belum terinstall, jalankan: pip install umap-learn")
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
    parser = argparse.ArgumentParser(
        description="Deteksi kandidat mislabel dataset anti-spoofing wajah menggunakan SigLIP."
    )
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT, help="Path folder dataset")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Nama model HuggingFace SigLIP")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Folder penyimpanan hasil analisis")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Ukuran batch ekstraksi")
    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.10,
        help="Proporsi outlier yang di-trim saat menghitung centroid kelas (0.0 - 0.4)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = collect_image_paths(args.dataset_root, CLASS_NAMES)
    if len(df) == 0:
        print(f"Tidak ada gambar ditemukan, periksa kembali dataset_root: {args.dataset_root}")
        return

    processor, model = load_model(args.model_name, DEVICE)
    df, embeddings = extract_embeddings(
        df, processor, model, cache_dir=args.output_dir, batch_size=args.batch_size, device=DEVICE
    )

    embeddings_path = os.path.join(args.output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Embedding disimpan di {embeddings_path}")

    centroids = compute_centroids(
        embeddings,
        df["label"].tolist(),
        CLASS_NAMES,
        md5_hashes=df["md5"].tolist() if "md5" in df.columns else None,
        trim_ratio=args.trim_ratio,
    )

    result = compute_similarity_and_margin(
        df, embeddings, centroids, CLASS_NAMES, use_loo_own_similarity=True
    )

    csv_path = os.path.join(args.output_dir, "hasil_deteksi_mislabel.csv")
    result.to_csv(csv_path, index=False)
    print(f"Hasil lengkap disimpan di {csv_path}")

    suspicious = result[result["margin"] < 0]
    suspicious_path = os.path.join(args.output_dir, "kandidat_mislabel.csv")
    suspicious.to_csv(suspicious_path, index=False)
    print(f"Ditemukan {len(suspicious)} kandidat mislabel, disimpan di {suspicious_path}")

    umap_path = os.path.join(args.output_dir, "umap_visualization.png")
    visualize_umap(embeddings, df["label"].tolist(), umap_path)

    print("Selesai. Silakan review manual file kandidat_mislabel.csv sebelum melakukan relabel final.")


if __name__ == "__main__":
    main()