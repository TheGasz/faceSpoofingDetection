"""
dataset_sorter.py
==================
Tool untuk mensortir ulang (relabeling) file dataset anti-spoofing wajah
berdasarkan CSV "label_sekarang / label_seharusnya".

Cara kerja singkat:
1. Ada 6 folder kategori: fake_mannequin, fake_mask, fake_printed,
   fake_screen, fake_unknown, realperson.
2. Setiap folder punya 1 CSV yang isinya: id/nama file, label saat ini
   (selalu sama dengan folder itu sendiri), dan label yang seharusnya
   (hasil review ulang).
3. Untuk setiap baris, jika label_seharusnya != label_sekarang, file
   tersebut dipindah (atau di-copy) ke folder yang sesuai dengan
   label_seharusnya.
4. Semua aksi (moved / copied / already_correct / not_found / skipped)
   dicatat ke report CSV supaya bisa diaudit / dry-run dulu sebelum
   benar-benar dieksekusi.

Contoh eksekusi CLI:
    python dataset_sorter.py --base_dir Convnext/train --csv_dir data/relabeling_csv --dry_run
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. KONFIGURASI LABEL -> (nama folder, prefix nama file)
# ---------------------------------------------------------------------------
LABEL_MAP: Dict[str, Tuple[str, str]] = {
    "MANNE":      ("fake_mannequin", "mannequin"),
    "MANNEQUIN":  ("fake_mannequin", "mannequin"),
    "MASK":       ("fake_mask",      "mask"),
    "S":          ("fake_screen",    "screen"),
    "SCREEN":     ("fake_screen",    "screen"),
    "P":          ("fake_printed",   "printed"),
    "PRINTED":    ("fake_printed",   "printed"),
    "U":          ("fake_unknown",   "unknown"),
    "UNKNOWN":    ("fake_unknown",   "unknown"),
    "REAL":       ("realperson",     "real"),
    "REALPERSON": ("realperson",     "real"),
}

CSV_CONFIGS = [
    {
        "csv_name": "FOLDER_FAKE_MANNE.CSV",
        "source_folder": "fake_mannequin",
        "id_col": "id",
        "current_col": "label_sekarang",
        "target_col": "label_seharusnya",
        "id_is_filename": False,
        "prefix": "mannequin",
    },
    {
        "csv_name": "FOLDER_FAKE_MASK.CSV",
        "source_folder": "fake_mask",
        "id_col": "id",
        "current_col": "label_sekarang",
        "target_col": "label seharusnya",  # ada spasi ekstra di header aslinya
        "id_is_filename": False,
        "prefix": "mask",
    },
    {
        "csv_name": "FOLDER_FAKE_SCREEN.CSV",
        "source_folder": "fake_screen",
        "id_col": "id",
        "current_col": "label_sekarang",
        "target_col": "label_seharusnya",
        "id_is_filename": False,
        "prefix": "screen",
    },
    {
        "csv_name": "relabeling_fake_printed.csv",
        "source_folder": "fake_printed",
        "id_col": "Id",
        "current_col": "Label Sebelumnya",
        "target_col": "Label Seharusnya",
        "id_is_filename": True,
        "prefix": "printed",
    },
    {
        "csv_name": "relabeling_fake_unknown.csv",
        "source_folder": "fake_unknown",
        "id_col": "Id",
        "current_col": "Label Sebelumnya",
        "target_col": "Label Seharusnya",
        "id_is_filename": True,
        "prefix": "unknown",
    },
    {
        "csv_name": "relabeling_realperson.csv",
        "source_folder": "realperson",
        "id_col": "Id",
        "current_col": "Label Sebelumnya",
        "target_col": "Label Seharusnya",
        "id_is_filename": True,
        "prefix": "real",
    },
]

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}


@dataclass
class ActionResult:
    csv_name: str
    row_id: str
    filename: Optional[str]
    source_folder: str
    current_label: str
    target_label: str
    target_folder: Optional[str]
    status: str            # moved / copied / already_correct / not_found / error / dry_run / skipped
    detail: str = ""
    target_prefix: Optional[str] = None
    final_filename: Optional[str] = None


def _clean_header(name: str) -> str:
    return name.strip()


def _normalize_label(label: str) -> str:
    return str(label).strip().upper()


def _resolve_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (folder_name, prefix) for a label code, or (None, None) if unknown."""
    key = _normalize_label(label)
    return LABEL_MAP.get(key, (None, None))


def _file_md5(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_filename_index(folder_path: str) -> Dict[Tuple[str, int], str]:
    """
    Scan a folder once and build an index:
        {(prefix_lower, numeric_id): actual_filename}
    so we can resolve "id-only" CSV rows (e.g. id=7 -> mannequin_007.jpg)
    without guessing extensions or zero-padding.
    """
    index: Dict[Tuple[str, int], str] = {}
    if not os.path.isdir(folder_path):
        return index

    pattern = re.compile(r"^(?P<prefix>[a-zA-Z]+)_(?P<num>\d+)\.(?P<ext>\w+)$")
    for fname in sorted(os.listdir(folder_path)):
        full = os.path.join(folder_path, fname)
        if not os.path.isfile(full):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        ext = m.group("ext").lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        key = (m.group("prefix").lower(), int(m.group("num")))
        index[key] = fname
    return index


def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        reader.fieldnames = [_clean_header(h) for h in reader.fieldnames]
        rows = []
        for row in reader:
            rows.append({
                _clean_header(k): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items() if k is not None
            })
        return rows


def build_tasks(base_dir: str, csv_dir: str) -> List[ActionResult]:
    """
    Read every configured CSV and turn it into a flat list of ActionResult.
    """
    tasks: List[ActionResult] = []

    for cfg in CSV_CONFIGS:
        csv_path = os.path.join(csv_dir, cfg["csv_name"])
        source_folder_path = os.path.join(base_dir, cfg["source_folder"])

        if not os.path.isfile(csv_path):
            print(f"[WARNING] CSV tidak ditemukan, dilewati: {csv_path}")
            continue

        rows = _read_csv_rows(csv_path)

        file_index = None
        if not cfg["id_is_filename"]:
            file_index = _build_filename_index(source_folder_path)

        for row in rows:
            raw_id = row.get(cfg["id_col"], "")
            current_label = row.get(cfg["current_col"], "")
            target_label = row.get(cfg["target_col"], "")

            filename = None
            if cfg["id_is_filename"]:
                filename = raw_id
            else:
                try:
                    numeric_id = int(str(raw_id).strip())
                except ValueError:
                    numeric_id = None
                if numeric_id is not None and file_index is not None:
                    filename = file_index.get((cfg["prefix"], numeric_id))

            target_folder, target_prefix = _resolve_label(target_label)

            if filename is None:
                tasks.append(ActionResult(
                    csv_name=cfg["csv_name"], row_id=str(raw_id), filename=None,
                    source_folder=cfg["source_folder"], current_label=current_label,
                    target_label=target_label, target_folder=target_folder,
                    status="not_found",
                    detail="File fisik tidak ditemukan di folder sumber (cek penamaan/ekstensi).",
                    target_prefix=target_prefix,
                ))
                continue

            if target_folder is None:
                tasks.append(ActionResult(
                    csv_name=cfg["csv_name"], row_id=str(raw_id), filename=filename,
                    source_folder=cfg["source_folder"], current_label=current_label,
                    target_label=target_label, target_folder=None,
                    status="error",
                    detail=f"Label tujuan '{target_label}' tidak dikenali di LABEL_MAP.",
                    target_prefix=target_prefix,
                ))
                continue

            tasks.append(ActionResult(
                csv_name=cfg["csv_name"], row_id=str(raw_id), filename=filename,
                source_folder=cfg["source_folder"], current_label=current_label,
                target_label=target_label, target_folder=target_folder,
                status="pending",
                target_prefix=target_prefix,
            ))

    return tasks


def _scan_existing_max(dst_dir: str, prefix: str) -> Tuple[int, int]:
    """
    Cari nomor urut tertinggi yang sudah dipakai file berpola
    '<prefix>_<angka>.<ext>' di dst_dir, plus lebar zero-padding yang dipakai.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.\w+$", re.IGNORECASE)
    max_num, width = 0, 3
    if os.path.isdir(dst_dir):
        for fname in os.listdir(dst_dir):
            m = pattern.match(fname)
            if m:
                num_str = m.group(1)
                num = int(num_str)
                if num > max_num:
                    max_num = num
                    width = len(num_str)
    return max_num, width


def execute_tasks(
    tasks: List[ActionResult],
    base_dir: str,
    mode: str = "move",
    dry_run: bool = True,
    on_conflict: str = "rename",
    rename_to_target_convention: bool = False,
    skip_exact_duplicates_in_target: bool = False,
) -> List[ActionResult]:
    """
    Execute the pending tasks: move/copy files whose target folder differs
    from their source folder.
    Pre-scans all target folders before moving any file to avoid cascade/ordering bugs.
    """
    assert mode in ("move", "copy")
    assert on_conflict in ("rename", "skip", "overwrite")

    # Pre-scan seluruh folder tujuan sebelum eksekusi agar urutan pemindahan tidak saling mempengaruhi
    counters: Dict[Tuple[str, str], List[int]] = {}
    target_hashes: Dict[str, Dict[str, str]] = {}

    unique_targets = {
        (os.path.join(base_dir, t.target_folder), t.target_prefix)
        for t in tasks if t.target_folder and t.target_prefix
    }
    for dst_dir, prefix in unique_targets:
        max_num, width = _scan_existing_max(dst_dir, prefix)
        counters[(dst_dir, prefix)] = [max_num + 1, width]
        if skip_exact_duplicates_in_target and os.path.isdir(dst_dir):
            target_hashes[dst_dir] = {}
            for fname in os.listdir(dst_dir):
                fpath = os.path.join(dst_dir, fname)
                if os.path.isfile(fpath):
                    target_hashes[dst_dir][_file_md5(fpath)] = fname

    for t in tasks:
        if t.status != "pending":
            continue

        if t.target_folder == t.source_folder:
            t.status = "already_correct"
            t.detail = "Label seharusnya sama dengan folder saat ini."
            t.final_filename = t.filename
            continue

        src_path = os.path.join(base_dir, t.source_folder, t.filename)
        dst_dir = os.path.join(base_dir, t.target_folder)

        if not os.path.isfile(src_path):
            t.status = "not_found"
            t.detail = f"File sumber hilang di disk: {src_path}"
            continue

        if skip_exact_duplicates_in_target:
            src_hash = _file_md5(src_path)
            existing_dup = target_hashes.get(dst_dir, {}).get(src_hash)
            if existing_dup:
                t.status = "skipped"
                t.final_filename = existing_dup
                t.detail = f"Duplikat MD5 identik sudah ada di folder tujuan: {existing_dup}"
                continue

        _, ext = os.path.splitext(t.filename)

        if rename_to_target_convention and t.target_prefix:
            key = (dst_dir, t.target_prefix)
            if key not in counters:
                max_num, width = _scan_existing_max(dst_dir, t.target_prefix)
                counters[key] = [max_num + 1, width]
            next_num, width = counters[key]
            new_filename = f"{t.target_prefix}_{next_num:0{width}d}{ext.lower()}"
            counters[key][0] += 1
            dst_path = os.path.join(dst_dir, new_filename)
        else:
            new_filename = t.filename
            dst_path = os.path.join(dst_dir, new_filename)

        if os.path.exists(dst_path) and not (rename_to_target_convention and t.target_prefix):
            if on_conflict == "skip":
                t.status = "skipped"
                t.detail = f"Sudah ada file dengan nama sama di {dst_path}"
                continue
            elif on_conflict == "rename":
                base, e = os.path.splitext(new_filename)
                i = 1
                new_path = os.path.join(dst_dir, f"{base}_dup{i}{e}")
                while os.path.exists(new_path):
                    i += 1
                    new_path = os.path.join(dst_dir, f"{base}_dup{i}{e}")
                dst_path = new_path
                new_filename = os.path.basename(dst_path)

        t.final_filename = new_filename

        if dry_run:
            t.status = "dry_run"
            t.detail = f"[DRY RUN] Akan {mode} '{t.filename}' -> {dst_path}"
            continue

        try:
            os.makedirs(dst_dir, exist_ok=True)
            if mode == "move":
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            t.status = "moved" if mode == "move" else "copied"
            t.detail = dst_path
        except Exception as e:
            t.status = "error"
            t.detail = str(e)

    return tasks


def write_report(tasks: List[ActionResult], report_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "csv_name", "row_id", "filename", "final_filename", "source_folder", "current_label",
            "target_label", "target_folder", "status", "detail",
        ])
        for t in tasks:
            writer.writerow([
                t.csv_name, t.row_id, t.filename, t.final_filename, t.source_folder, t.current_label,
                t.target_label, t.target_folder, t.status, t.detail,
            ])


def print_summary(tasks: List[ActionResult]):
    counts = Counter(t.status for t in tasks)
    print("\n=== RINGKASAN ===")
    for status, n in sorted(counts.items()):
        print(f"  {status:16s}: {n}")
    print(f"  {'TOTAL':16s}: {len(tasks)}")


def run_sorting(
    base_dir: str,
    csv_dir: str,
    mode: str = "move",
    dry_run: bool = True,
    on_conflict: str = "rename",
    report_path: Optional[str] = None,
    rename_to_target_convention: bool = False,
    skip_exact_duplicates_in_target: bool = False,
) -> List[ActionResult]:
    """
    Fungsi utama untuk dipanggil dari notebook / script lain.
    Mengembalikan list[ActionResult].
    """
    tasks = build_tasks(base_dir, csv_dir)
    tasks = execute_tasks(
        tasks,
        base_dir,
        mode=mode,
        dry_run=dry_run,
        on_conflict=on_conflict,
        rename_to_target_convention=rename_to_target_convention,
        skip_exact_duplicates_in_target=skip_exact_duplicates_in_target,
    )
    print_summary(tasks)
    if report_path:
        write_report(tasks, report_path)
        print(f"\nReport tersimpan di: {report_path}")
    return tasks


def _main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base = os.path.join(script_dir, "train")
    default_csv = os.path.join(script_dir, "data", "relabeling_csv")
    default_report = os.path.join(default_csv, "sorting_report.csv")

    parser = argparse.ArgumentParser(description="Sortir ulang dataset berdasarkan CSV relabeling.")
    parser.add_argument("--base_dir", default=default_base, help="Folder induk yang berisi 6 folder kategori.")
    parser.add_argument("--csv_dir", default=default_csv, help="Folder tempat 6 file CSV berada.")
    parser.add_argument("--mode", choices=["move", "copy"], default="move")
    parser.add_argument("--dry_run", action="store_true", help="Preview saja, tidak mengubah file apapun.")
    parser.add_argument("--on_conflict", choices=["rename", "skip", "overwrite"], default="rename")
    parser.add_argument("--report", default=None, help="Path file report CSV (opsional).")
    parser.add_argument(
        "--rename_to_target_convention",
        action="store_true",
        help="Ganti nama file mengikuti prefix folder tujuan + nomor urut berikutnya.",
    )
    parser.add_argument(
        "--skip_exact_duplicates_in_target",
        action="store_true",
        help="Lewati pemindahan jika file dengan MD5 identik sudah ada di folder tujuan.",
    )
    args = parser.parse_args()

    report_path = args.report if args.report is not None else (None if args.dry_run else default_report)

    run_sorting(
        base_dir=args.base_dir,
        csv_dir=args.csv_dir,
        mode=args.mode,
        dry_run=args.dry_run,
        on_conflict=args.on_conflict,
        report_path=report_path,
        rename_to_target_convention=args.rename_to_target_convention,
        skip_exact_duplicates_in_target=args.skip_exact_duplicates_in_target,
    )


if __name__ == "__main__":
    _main()
