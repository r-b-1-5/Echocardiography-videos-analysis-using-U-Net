"""EchoNet-Dynamic CSV/video loading, trace-mask creation, and paired ED/ES data."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


def locate_dataset_root(data_root_override: Optional[str] = None) -> Tuple[Path, Path, Path]:
    """Find the directory containing both FileList.csv and VolumeTracings.csv."""
    if data_root_override is not None:
        base = Path(data_root_override).expanduser()
        if not base.exists():
            raise FileNotFoundError(f"DATA_ROOT_OVERRIDE does not exist: {base}")
        candidates = list(base.rglob("FileList.csv"))
    else:
        search_roots = [Path("/kaggle/input"), Path("/content"), Path.cwd()]
        candidates = []
        for search_root in search_roots:
            if search_root.exists():
                try:
                    candidates.extend(search_root.rglob("FileList.csv"))
                except PermissionError:
                    pass
        candidates = list(dict.fromkeys(candidates))

    for filelist_path in candidates:
        root = filelist_path.parent
        if (root / "VolumeTracings.csv").exists():
            return root, filelist_path, root / "VolumeTracings.csv"

    listed = "\n".join(str(path) for path in candidates[:20]) or "(no FileList.csv files found)"
    raise FileNotFoundError(
        "Could not locate a folder containing BOTH FileList.csv and VolumeTracings.csv.\n"
        "Check that the EchoNet-Dynamic dataset has been added to this environment.\n"
        f"FileList candidates found:\n{listed}"
    )


def find_column(df: pd.DataFrame, candidates, required: bool = True):
    """Find a CSV column without depending on capitalization."""
    lower_map = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}. Found: {df.columns.tolist()}"
        )
    return None


def normalize_stem(value) -> str:
    return Path(str(value).strip()).stem


def trace_rows_to_array(group: pd.DataFrame, columns: Dict[str, str]) -> np.ndarray:
    # Preserve CSV row order: the first line is the long axis; subsequent lines are short axes.
    return group[[columns["x1"], columns["y1"], columns["x2"], columns["y2"]]].to_numpy(dtype=np.float32)


def trace_polygon_mask(trace_array, height: int, width: int):
    """Rasterize LV cavity polygon using short-axis endpoints (official EchoNet convention)."""
    if trace_array is None or len(trace_array) < 3:
        return None
    x = np.concatenate([trace_array[1:, 0], np.flip(trace_array[1:, 2])])
    y = np.concatenate([trace_array[1:, 1], np.flip(trace_array[1:, 3])])
    points = np.stack([np.rint(x), np.rint(y)], axis=1).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 1)
    return mask


def read_gray_frame(cap, frame_index: int, image_size: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape != (image_size, image_size):
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)


def get_ef_label(row, ef_col, edv_col, esv_col) -> float:
    ef = np.nan
    if ef_col is not None:
        ef = pd.to_numeric(pd.Series([row[ef_col]]), errors="coerce").iloc[0]
    if pd.isna(ef) and edv_col is not None and esv_col is not None:
        edv = pd.to_numeric(pd.Series([row[edv_col]]), errors="coerce").iloc[0]
        esv = pd.to_numeric(pd.Series([row[esv_col]]), errors="coerce").iloc[0]
        if pd.notna(edv) and pd.notna(esv) and edv > 0:
            ef = 100.0 * (edv - esv) / edv
    if pd.isna(ef):
        return np.nan
    # Some mirrors may store EF as a fraction rather than a percentage.
    if 0 <= float(ef) <= 1.0:
        ef = float(ef) * 100.0
    return float(ef)


def prepare_splits(config) -> Dict[str, dict]:
    """Read source files and build paired ED/ES images, masks, EF labels, and metadata."""
    data_root, filelist_path, traces_path = locate_dataset_root(config.data_root)
    print("Dataset location found:")
    print("  Dataset root      :", data_root)
    print("  FileList.csv      :", filelist_path)
    print("  VolumeTracings.csv:", traces_path)

    filelist_df = pd.read_csv(filelist_path)
    traces_df = pd.read_csv(traces_path)
    print("\nFileList.csv inspection")
    print("  Shape   :", filelist_df.shape)
    print("  Columns :", filelist_df.columns.tolist())
    print(filelist_df.head(5).to_string(index=False))
    print("\nVolumeTracings.csv inspection")
    print("  Shape   :", traces_df.shape)
    print("  Columns :", traces_df.columns.tolist())
    print(traces_df.head(8).to_string(index=False))

    file_col = find_column(filelist_df, ["FileName", "filename", "video", "video_name"])
    split_col = find_column(filelist_df, ["Split", "split"])
    ef_col = find_column(filelist_df, ["EF", "EjectionFraction", "Ejection_Fraction"], required=False)
    edv_col = find_column(filelist_df, ["EDV", "EndDiastolicVolume"], required=False)
    esv_col = find_column(filelist_df, ["ESV", "EndSystolicVolume"], required=False)
    trace_file_col = find_column(traces_df, ["FileName", "filename"])
    trace_frame_col = find_column(traces_df, ["Frame", "frame"])
    trace_columns = {
        "x1": find_column(traces_df, ["X1", "x1"]),
        "y1": find_column(traces_df, ["Y1", "y1"]),
        "x2": find_column(traces_df, ["X2", "x2"]),
        "y2": find_column(traces_df, ["Y2", "y2"]),
    }
    print("\nColumn mapping confirmed:")
    print("  Study filename :", file_col)
    print("  Official split :", split_col)
    print("  EF label       :", ef_col)
    print("  EDV label      :", edv_col)
    print("  ESV label      :", esv_col)
    print("  Trace filename/frame and X1,Y1,X2,Y2 columns found.")

    split_aliases = {
        "TRAIN": "train", "TRAINING": "train",
        "VAL": "val", "VALID": "val", "VALIDATION": "val",
        "TEST": "test", "TESTING": "test",
    }
    filelist_df = filelist_df.copy()
    filelist_df["_key"] = filelist_df[file_col].astype(str).map(normalize_stem)
    filelist_df["_split"] = (
        filelist_df[split_col].astype(str).str.strip().str.upper().map(split_aliases)
    )
    print("Original split values:", sorted(filelist_df[split_col].astype(str).str.strip().unique().tolist()))
    print("\nRows by recognized official split:")
    print(filelist_df["_split"].value_counts(dropna=False).to_string())
    unknown_splits = (
        filelist_df.loc[filelist_df["_split"].isna(), split_col].astype(str).unique().tolist()
    )
    if unknown_splits:
        print("\nWARNING: unrecognized split labels will be ignored:", unknown_splits)

    video_dir = data_root / "Videos"
    video_paths = sorted(video_dir.rglob("*.mp4"))
    video_map = {path.stem: path for path in video_paths}
    print("Video directory:", video_dir)
    print(f"MP4 videos found: {len(video_paths):,}")
    for path in video_paths[:5]:
        print(path)
    if not video_paths:
        raise FileNotFoundError(f"No AVI videos found in {video_dir}")

    traces_df = traces_df.copy()
    traces_df["_key"] = traces_df[trace_file_col].astype(str).map(normalize_stem)
    traces_df["_frame"] = pd.to_numeric(traces_df[trace_frame_col], errors="coerce")
    trace_frame_counts = traces_df.dropna(subset=["_frame"]).groupby("_key")["_frame"].nunique()
    print("\nTrace coverage summary:")
    print("  Unique studies with traces :", traces_df["_key"].nunique())
    print("  Studies with >=2 trace frames:", int((trace_frame_counts >= 2).sum()))
    print("  Studies with <2 trace frames :", int((trace_frame_counts < 2).sum()))

    inspect_key = next((
        key for key in filelist_df["_key"]
        if key in video_map and key in trace_frame_counts.index and trace_frame_counts.get(key, 0) >= 2
    ), None)
    if inspect_key is None:
        raise RuntimeError(
            "No study has both a video and at least two traced frames. "
            "Check file naming and the dataset structure."
        )
    inspect_path = video_map[inspect_key]
    cap = cv2.VideoCapture(str(inspect_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, inspect_frame = cap.read()
    cap.release()
    print("\nExample video inspection:")
    print("  File key  :", inspect_key)
    print("  Path      :", inspect_path)
    print(f"  Dimensions: {width} × {height}")
    print(f"  Frames    : {frame_count}")
    print(f"  FPS       : {fps:.2f}")
    print("  First frame read successfully:", bool(ret))
    print(
        "  Trace frames:",
        sorted(traces_df.loc[traces_df["_key"] == inspect_key, "_frame"].dropna().astype(int).unique().tolist()),
    )
    if ret:
        fig, ax = plt.subplots(figsize=(4, 4))
        if inspect_frame.ndim == 3:
            inspect_gray = cv2.cvtColor(inspect_frame, cv2.COLOR_BGR2GRAY)
        else:
            inspect_gray = inspect_frame
        ax.imshow(inspect_gray, cmap="gray")
        ax.set_title("Raw example video frame")
        ax.axis("off")
        fig.savefig(config.resolved_output_dir / "example_raw_video_frame.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    split_limits = {
        "train": config.train_limit,
        "val": config.val_limit,
        "test": config.test_limit,
    }
    selected_rows = {}
    for split_name in ["train", "val", "test"]:
        split_rows = filelist_df[filelist_df["_split"] == split_name].copy()
        limit = split_limits[split_name]
        if limit is not None:
            if int(limit) < 0:
                raise ValueError(f"{split_name} limit must be >=0 or None, got {limit}")
            if int(limit) < len(split_rows):
                split_rows = split_rows.sample(n=int(limit), random_state=config.seed)
        selected_rows[split_name] = split_rows.reset_index(drop=True)
        print(
            f"{split_name.upper():5s}: selected {len(split_rows):,} / "
            f"{int((filelist_df['_split'] == split_name).sum()):,} official rows"
        )

    if any(len(selected_rows[name]) == 0 for name in ["train", "val", "test"]):
        raise ValueError(
            "Train, validation, and test splits must all contain at least one selected study. "
            "Set train/val/test limits appropriately."
        )

    trace_groups = defaultdict(list)
    for (key, frame), group in traces_df.dropna(subset=["_frame"]).groupby(
        ["_key", "_frame"], sort=False
    ):
        trace_groups[key].append((int(frame), trace_rows_to_array(group, trace_columns)))

    prepared = {
        split: {"images": [], "masks": [], "ef": [], "meta": []}
        for split in ["train", "val", "test"]
    }
    skip_reasons = defaultdict(int)

    for split_name in ["train", "val", "test"]:
        rows = selected_rows[split_name]
        print(f"\nPreparing {split_name.upper()} split ({len(rows):,} requested studies)...")
        for row_idx, (_, row) in enumerate(rows.iterrows(), start=1):
            key = row["_key"]
            if key not in video_map:
                skip_reasons["video_missing"] += 1
                continue
            frame_groups = trace_groups.get(key, [])
            if len(frame_groups) < 2:
                skip_reasons["fewer_than_two_traced_frames"] += 1
                continue
            ef = get_ef_label(row, ef_col, edv_col, esv_col)
            if not np.isfinite(ef):
                skip_reasons["missing_EF_label"] += 1
                continue

            cap = cv2.VideoCapture(str(video_map[key]))
            candidates = []
            try:
                for frame_index, trace_array in frame_groups:
                    gray = read_gray_frame(cap, frame_index, config.image_size)
                    if gray is None:
                        continue
                    # Coordinates in the release correspond to the standardized frame grid.
                    # Use the original video frame shape before resize if it is not 112×112.
                    # EchoNet-Dynamic is normally 112×112; for nonstandard mirrors we resize
                    # the mask to the model grid after rasterizing at the original dimensions.
                    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    raw_mask = trace_polygon_mask(trace_array, orig_h, orig_w)
                    if raw_mask is None:
                        continue
                    if raw_mask.shape != (config.image_size, config.image_size):
                        raw_mask = cv2.resize(
                            raw_mask, (config.image_size, config.image_size),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    area = int(raw_mask.sum())
                    if area > 0:
                        candidates.append((area, frame_index, gray, raw_mask, trace_array))
            finally:
                cap.release()

            if len(candidates) < 2:
                skip_reasons["could_not_decode_two_annotated_frames"] += 1
                continue
            candidates.sort(key=lambda item: item[0])
            es = candidates[0]
            ed = candidates[-1]
            if ed[1] == es[1]:
                skip_reasons["same_ED_ES_frame"] += 1
                continue

            image_pair = np.stack([ed[2], es[2]], axis=0).astype(np.uint8)
            mask_pair = np.stack([ed[3], es[3]], axis=0).astype(np.uint8)
            prepared[split_name]["images"].append(image_pair)
            prepared[split_name]["masks"].append(mask_pair)
            prepared[split_name]["ef"].append(float(ef))
            long_axis_ed = (
                float(np.linalg.norm(ed[4][0, 2:4] - ed[4][0, 0:2])) if len(ed[4]) else np.nan
            )
            prepared[split_name]["meta"].append({
                "FileName": str(row[file_col]),
                "key": key,
                "split": split_name,
                "EF_clinical": float(ef),
                "ED_frame": int(ed[1]),
                "ES_frame": int(es[1]),
                "ED_mask_area_px": int(ed[0]),
                "ES_mask_area_px": int(es[0]),
                "long_axis_ED_px_from_trace": long_axis_ed,
            })
            if row_idx % 500 == 0 or row_idx == len(rows):
                print(
                    f"  Processed {row_idx:,}/{len(rows):,} rows | "
                    f"usable pairs so far: {len(prepared[split_name]['images']):,}"
                )

        print(
            f"  Completed {split_name.upper()}: "
            f"{len(prepared[split_name]['images']):,} paired studies ready"
        )

    print("\nPreprocessing skip-reason counts:")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            print(f"  {reason}: {count:,}")
    else:
        print("  No skipped studies.")

    for split_name in ["train", "val", "test"]:
        count = len(prepared[split_name]["images"])
        if count == 0:
            raise RuntimeError(f"No usable studies in {split_name}. Check annotations and video paths.")
        prepared[split_name]["images"] = np.stack(prepared[split_name]["images"])
        prepared[split_name]["masks"] = np.stack(prepared[split_name]["masks"])
        prepared[split_name]["ef"] = np.asarray(prepared[split_name]["ef"], dtype=np.float32)
        prepared[split_name]["meta"] = pd.DataFrame(prepared[split_name]["meta"])
        print(
            f"{split_name.upper():5s} arrays: "
            f"images={prepared[split_name]['images'].shape}, "
            f"masks={prepared[split_name]['masks'].shape}, "
            f"EF={prepared[split_name]['ef'].shape}"
        )

    print("\nFinal usable study totals (after missing-file/trace filtering):")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name.upper():5s}: {len(prepared[split_name]['ef']):,}")
    return prepared


class EchoNetPairs(Dataset):
    def __init__(self, split_data, augment: bool = False):
        self.images = split_data["images"]
        self.masks = split_data["masks"]
        self.ef = split_data["ef"]
        self.meta = split_data["meta"].reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.ef)

    def __getitem__(self, idx):
        # uint8 arrays keep the in-memory cache compact; convert to float on access.
        x = self.images[idx].astype(np.float32) / 255.0
        y = self.masks[idx].astype(np.int64)
        if self.augment and random.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(self.ef[idx], dtype=torch.float32),
        )


def build_dataloaders(prepared: Dict[str, dict], config, device: torch.device):
    train_ds = EchoNetPairs(prepared["train"], augment=True)
    val_ds = EchoNetPairs(prepared["val"], augment=False)
    test_ds = EchoNetPairs(prepared["test"], augment=False)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=pin_memory,
    )
    x_batch, y_batch, ef_batch = next(iter(train_loader))
    print("Training batch sanity check:")
    print("  Images tensor:", tuple(x_batch.shape), x_batch.dtype,
          f"range={x_batch.min().item():.3f}..{x_batch.max().item():.3f}")
    print("  Masks tensor :", tuple(y_batch.shape), y_batch.dtype,
          f"labels={torch.unique(y_batch).tolist()}")
    print("  EF tensor    :", tuple(ef_batch.shape), ef_batch.dtype,
          f"range={ef_batch.min().item():.1f}..{ef_batch.max().item():.1f}%")
    return (train_ds, val_ds, test_ds), (train_loader, val_loader, test_loader)
