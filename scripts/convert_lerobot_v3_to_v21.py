"""Convert a LeRobot v3.0 dataset back to v2.1 format.

WHY THIS EXISTS
---------------
openpi pins lerobot at an old commit that only reads v2.1 datasets, but
`lerobot-record` (any recent version) writes v3.0. Without this script you'd
have to either re-record everything or upgrade the entire openpi+jax dep stack.

WHERE TO RUN
------------
Run on a machine that has:
  - Python >= 3.10
  - A recent `lerobot` install (anything that can READ v3.0 — i.e. v0.4.0+)
  - ffmpeg in PATH

That's your local laptop, the one you recorded with. NOT inside openpi's `.venv`
on Spartan (that one has the OLD lerobot which can't read v3.0).

WHAT IT DOES
------------
1. Loads the v3.0 dataset (many episodes packed into chunked parquet/mp4 files).
2. For each episode:
     - filters the v3 parquet rows for that episode -> writes
       `data/chunk-000/episode_XXXXXX.parquet`
     - re-encodes the per-episode video segment for each camera (AV1 -> H.264)
       to `videos/chunk-000/<camera_key>/episode_XXXXXX.mp4`
     - computes per-episode stats (min/max/mean/std/count)
3. Writes v2.1 metadata: info.json, episodes.jsonl, tasks.jsonl,
   episodes_stats.jsonl.
4. Optionally pushes to HF Hub as a new dataset repo.

USAGE
-----
    pip install lerobot pyarrow pandas tqdm  # ffmpeg from your package mgr

    python scripts/convert_lerobot_v3_to_v21.py \\
        --src-repo-id LUOSYrrrrr/so101_yellow_tape_v1 \\
        --dst-repo-id LUOSYrrrrr/so101_yellow_tape_v1_v21 \\
        --out-dir ~/.cache/huggingface/lerobot/LUOSYrrrrr/so101_yellow_tape_v1_v21 \\
        --push-to-hub

LIMITATIONS
-----------
- Targets a single-chunk dataset (chunk-000). Datasets with > ~1000 episodes
  may need chunk-NNN handling.
- Re-encodes video to libx264 yuv420p (smaller, universally supported, fast).
- Only handles `dtype=video` cameras and float/int low-dim features.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


# v2.1 path templates
DATA_PATH_V21 = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_V21 = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def download_v3_dataset(repo_id: str, local_dir: Path) -> Path:
    """Pull the entire v3 dataset to a local cache dir (idempotent)."""
    print(f"==> Downloading v3 dataset {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return local_dir


def load_v3_info(src_root: Path) -> dict:
    return json.loads((src_root / "meta" / "info.json").read_text())


def load_v3_episodes_df(src_root: Path) -> pd.DataFrame:
    """The v3 episodes parquet tells us which file each episode lives in,
    its frame range, and (optionally) per-episode stats columns.

    v3 sometimes SPLITS this into multiple parquets (one with metadata,
    another with stats). We merge by episode_index instead of stacking rows.
    """
    ep_files = sorted((src_root / "meta" / "episodes").glob("chunk-*/file*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episodes parquet under {src_root / 'meta' / 'episodes'}")

    dfs = [pd.read_parquet(p) for p in ep_files]
    # Print schema for diagnostics.
    for p, df in zip(ep_files, dfs):
        print(f"   {p.name}: {len(df)} rows, {len(df.columns)} cols")

    # Detect whether files are SHARDED (same columns, different episodes) or
    # SPLIT by content (same episodes, different columns). We check whether
    # the first column set fully matches the second.
    if len(dfs) == 1:
        return dfs[0]

    same_cols = all(set(d.columns) == set(dfs[0].columns) for d in dfs[1:])
    if same_cols:
        return pd.concat(dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)

    # Otherwise: merge on episode_index (outer join) so we get all columns.
    if not all("episode_index" in d.columns for d in dfs):
        raise ValueError(
            "Episode parquets have different column sets but at least one is missing "
            "'episode_index' so they can't be joined."
        )
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="episode_index", how="outer", suffixes=("", "_dup"))
        # Drop suffixed duplicates (right-side wins is fine since values match).
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols)
    return merged.sort_values("episode_index").reset_index(drop=True)


def load_v3_tasks(src_root: Path) -> pd.DataFrame:
    """Return a DataFrame with explicit `task` + `task_index` columns.

    v3 stores tasks in two possible shapes:
      - tasks.parquet indexed by task string with a `task_index` column
      - tasks.jsonl with `{task_index, task}` rows
    Normalize both to flat columns so downstream code is uniform.
    """
    tasks_path = src_root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        df = pd.read_parquet(tasks_path)
        if "task" not in df.columns:
            # The task string is in the index. Promote to a column.
            df = df.reset_index().rename(columns={df.index.name or "index": "task"})
        return df
    tasks_jsonl = src_root / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        rows = [json.loads(l) for l in tasks_jsonl.read_text().splitlines() if l.strip()]
        return pd.DataFrame(rows)
    raise FileNotFoundError(f"Neither tasks.parquet nor tasks.jsonl under {src_root / 'meta'}")


def load_v3_episode_data(src_root: Path, info: dict, episode_row: pd.Series) -> pd.DataFrame:
    """Open the v3 parquet that contains this episode and slice its rows."""
    chunk = int(episode_row["data/chunk_index"])
    file_idx = int(episode_row["data/file_index"])
    rel = info["data_path"].format(chunk_index=chunk, file_index=file_idx)
    df = pd.read_parquet(src_root / rel)
    ep_idx = int(episode_row["episode_index"])
    return df[df["episode_index"] == ep_idx].reset_index(drop=True)


def extract_episode_stats_from_row(episode_row: pd.Series, info: dict) -> dict:
    """Pull per-episode stats directly from the v3 episodes parquet columns
    (stored as `stats/<feat>/{min,max,mean,std,count,q01,...}`)."""
    stats: dict[str, dict] = {}
    for feat_name, feat_meta in info["features"].items():
        if feat_meta.get("dtype") == "video":
            continue
        prefix = f"stats/{feat_name}/"
        feat_cols = [c for c in episode_row.index if c.startswith(prefix)]
        if not feat_cols:
            continue
        s: dict[str, list] = {}
        for c in feat_cols:
            stat_name = c[len(prefix):]
            val = episode_row[c]
            if isinstance(val, np.ndarray):
                s[stat_name] = val.tolist()
            elif isinstance(val, list):
                s[stat_name] = val
            else:
                s[stat_name] = [float(val)]
        stats[feat_name] = s
    return stats


def ffmpeg_extract_episode_video(
    src_video: Path,
    dst_video: Path,
    from_timestamp: float,
    to_timestamp: float,
) -> None:
    """Re-encode the [t0, t1) segment to H.264 (frame-accurate cut)."""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    duration = to_timestamp - from_timestamp
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{from_timestamp:.6f}",
        "-i", str(src_video),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-an",
        str(dst_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def compute_episode_stats(ep_df: pd.DataFrame, info: dict) -> dict:
    """Per-feature min/max/mean/std/count for the numeric (non-video) features."""
    stats: dict[str, dict] = {}
    for feat_name, feat_meta in info["features"].items():
        if feat_meta.get("dtype") == "video":
            continue
        if feat_name not in ep_df.columns:
            continue
        col = ep_df[feat_name]
        # Convert object-of-arrays to a 2D numpy array for vector features.
        try:
            arr = np.asarray(col.to_list(), dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if arr.ndim == 1:
            arr = arr[:, None]
        stats[feat_name] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def write_v21_info(out_root: Path, src_info: dict, n_episodes: int) -> None:
    """Build a minimal v2.1 info.json from the v3 info."""
    info_v21: dict = {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "unknown"),
        "total_episodes": n_episodes,
        "total_frames": src_info["total_frames"],
        "total_tasks": src_info["total_tasks"],
        "total_videos": n_episodes * sum(
            1 for f in src_info["features"].values() if f.get("dtype") == "video"
        ),
        "total_chunks": 1,
        "chunks_size": max(n_episodes, 1000),
        "fps": src_info["fps"],
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": DATA_PATH_V21,
        "video_path": VIDEO_PATH_V21,
        "features": {},
    }
    # v2.1 features: drop the video-codec sub-fields that v3 added under "info".
    for name, meta in src_info["features"].items():
        if meta.get("dtype") == "video":
            v_info = meta.get("info", {})
            info_v21["features"][name] = {
                "dtype": "video",
                "shape": meta["shape"],
                "names": meta.get("names"),
                "info": {
                    "video.height": v_info.get("video.height"),
                    "video.width": v_info.get("video.width"),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": v_info.get("video.is_depth_map", False),
                    "video.fps": v_info.get("video.fps", src_info["fps"]),
                    "video.channels": v_info.get("video.channels", 3),
                    "has_audio": False,
                },
            }
        else:
            info_v21["features"][name] = meta

    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "info.json").write_text(json.dumps(info_v21, indent=4))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def convert(
    src_repo_id: str,
    dst_repo_id: str,
    out_dir: Path,
    src_cache: Path | None = None,
    push_to_hub: bool = False,
) -> None:
    if src_cache is None:
        src_cache = Path("/tmp") / src_repo_id.replace("/", "__")
    src_cache.mkdir(parents=True, exist_ok=True)
    download_v3_dataset(src_repo_id, src_cache)

    info = load_v3_info(src_cache)
    if info["codebase_version"] != "v3.0":
        raise ValueError(f"Expected codebase_version=v3.0, got {info['codebase_version']}")

    episodes_df = load_v3_episodes_df(src_cache)
    tasks_df = load_v3_tasks(src_cache)
    n_episodes = len(episodes_df)

    print(f"==> Source: {n_episodes} episodes, {info['total_frames']} frames, fps={info['fps']}")
    print(f"==> Output: {out_dir}")

    if out_dir.exists():
        print(f"   (cleaning existing output dir)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Sort by episode_index just in case.
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)

    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    episodes_jsonl: list[dict] = []
    episodes_stats_jsonl: list[dict] = []

    for _, ep in tqdm(episodes_df.iterrows(), total=n_episodes, desc="Converting episodes"):
        ep_idx = int(ep["episode_index"])
        ep_df = load_v3_episode_data(src_cache, info, ep)

        # ---- 1. Per-episode parquet ------------------------------------- #
        parquet_path = out_dir / DATA_PATH_V21.format(episode_chunk=0, episode_index=ep_idx)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(parquet_path)

        # ---- 2. Per-episode video clips --------------------------------- #
        # v3 stores per-camera chunk/file/from_timestamp/to_timestamp because
        # different cameras may end up in different concat files when sized
        # independently.
        for vk in video_keys:
            v_chunk = int(ep[f"videos/{vk}/chunk_index"])
            v_file = int(ep[f"videos/{vk}/file_index"])
            t0 = float(ep[f"videos/{vk}/from_timestamp"])
            t1 = float(ep[f"videos/{vk}/to_timestamp"])
            src_video_rel = info["video_path"].format(
                video_key=vk, chunk_index=v_chunk, file_index=v_file
            )
            src_video = src_cache / src_video_rel
            dst_video = out_dir / VIDEO_PATH_V21.format(
                episode_chunk=0, video_key=vk, episode_index=ep_idx
            )
            ffmpeg_extract_episode_video(src_video, dst_video, t0, t1)

        # ---- 3. Per-episode metadata + stats ---------------------------- #
        # Tasks for this episode: v3 stores a list of task strings per episode.
        tasks_field = ep.get("tasks")
        if isinstance(tasks_field, np.ndarray):
            tasks_list = tasks_field.tolist()
        elif isinstance(tasks_field, list):
            tasks_list = tasks_field
        else:
            tasks_list = [tasks_df.iloc[0]["task"]]  # fallback: single-task dataset

        episodes_jsonl.append({
            "episode_index": ep_idx,
            "tasks": tasks_list,
            "length": int(ep["length"]),
        })

        # Pull pre-computed per-episode stats out of the v3 episodes parquet
        # (columns like `stats/action/min`, `stats/action/q01`, etc.). Falls
        # back to recomputing from the raw frames if not present.
        ep_stats = extract_episode_stats_from_row(ep, info)
        if not ep_stats:
            ep_stats = compute_episode_stats(ep_df, info)
        episodes_stats_jsonl.append({"episode_index": ep_idx, "stats": ep_stats})

    # ---- 4. v2.1 meta files --------------------------------------------- #
    write_v21_info(out_dir, info, n_episodes)
    write_jsonl(out_dir / "meta" / "episodes.jsonl", episodes_jsonl)
    write_jsonl(out_dir / "meta" / "episodes_stats.jsonl", episodes_stats_jsonl)
    write_jsonl(
        out_dir / "meta" / "tasks.jsonl",
        [{"task_index": int(r["task_index"]), "task": r["task"]} for _, r in tasks_df.iterrows()],
    )

    print(f"==> Done. v2.1 dataset at {out_dir}")
    print(f"    Episodes: {n_episodes}")
    print(f"    Total frames: {info['total_frames']}")

    # ---- 5. Optional: push to HF Hub ------------------------------------ #
    if push_to_hub:
        print(f"==> Creating repo {dst_repo_id} (if not exists)...")
        api = HfApi()
        api.create_repo(repo_id=dst_repo_id, repo_type="dataset", exist_ok=True)
        print(f"==> Uploading to {dst_repo_id}...")
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=dst_repo_id,
            repo_type="dataset",
            commit_message="convert from v3.0 -> v2.1",
        )
        print(f"==> Uploaded: https://huggingface.co/datasets/{dst_repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-repo-id", required=True, help="Source v3.0 dataset on HF Hub")
    p.add_argument("--dst-repo-id", required=True, help="Target v2.1 dataset repo id")
    p.add_argument("--out-dir", required=True, type=Path, help="Local output dir for v2.1 files")
    p.add_argument("--src-cache", type=Path, default=None, help="Where to cache the downloaded v3 files (default: /tmp/...)")
    p.add_argument("--push-to-hub", action="store_true", help="Push converted dataset to HF Hub")
    args = p.parse_args()

    convert(
        src_repo_id=args.src_repo_id,
        dst_repo_id=args.dst_repo_id,
        out_dir=args.out_dir.expanduser(),
        src_cache=args.src_cache.expanduser() if args.src_cache else None,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
