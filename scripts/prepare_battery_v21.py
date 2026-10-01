"""Create a local LeRobot v2.1 view of the 100 successful battery episodes.

The source v3 dataset is left untouched. Videos are hard-linked where possible,
so the compatibility view consumes almost no additional video storage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from convert_lerobot_v3_to_v21 import (
    DATA_PATH_V21,
    VIDEO_PATH_V21,
    compute_episode_stats,
    load_v3_episode_data,
    load_v3_episodes_df,
    load_v3_info,
    load_v3_tasks,
    write_jsonl,
    write_v21_info,
)


SOURCE_REVISION = "ece8d17141f5138fa141677b2516991b99190ac7"
SOURCE = Path("/data/gpfs/projects/punim2341/siyuanluo/so_arm101/datasets/so101_battery_insertion_v2")
OUTPUT = Path(
    "/data/gpfs/projects/punim2341/siyuanluo/lerobot_cache/"
    "LUOSYrrrrr/so101_battery_insertion_v2_openpi_v21"
)
SELECTION = SOURCE / "meta/act_success_episodes.json"


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    selected = json.loads(SELECTION.read_text())
    if isinstance(selected, dict):
        selected = selected.get("episodes", selected.get("success_episodes"))
    selected = [int(index) for index in selected]
    if len(selected) != 100 or len(set(selected)) != 100:
        raise ValueError(f"Expected 100 unique successful episodes, got {len(selected)}")

    info = load_v3_info(SOURCE)
    episodes = load_v3_episodes_df(SOURCE).set_index("episode_index", drop=False)
    tasks = load_v3_tasks(SOURCE)
    missing = sorted(set(selected) - set(episodes.index))
    if missing:
        raise ValueError(f"Selected episodes missing from source: {missing}")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    video_keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    episode_rows = []
    stats_rows = []
    mapping = []
    global_index = 0

    for new_index, source_index in enumerate(tqdm(selected, desc="Preparing v2.1 episodes")):
        episode = episodes.loc[source_index]
        frames = load_v3_episode_data(SOURCE, info, episode)
        length = len(frames)
        if length != int(episode["length"]):
            raise ValueError(f"Episode {source_index}: expected {episode['length']} frames, got {length}")

        frames = frames.copy()
        frames["episode_index"] = new_index
        frames["frame_index"] = np.arange(length, dtype=np.int64)
        frames["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
        parquet_path = OUTPUT / DATA_PATH_V21.format(episode_chunk=0, episode_index=new_index)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frames.to_parquet(parquet_path, index=False)

        for video_key in video_keys:
            chunk = int(episode[f"videos/{video_key}/chunk_index"])
            file_index = int(episode[f"videos/{video_key}/file_index"])
            source_video = SOURCE / info["video_path"].format(
                video_key=video_key,
                chunk_index=chunk,
                file_index=file_index,
            )
            destination = OUTPUT / VIDEO_PATH_V21.format(
                episode_chunk=0,
                video_key=video_key,
                episode_index=new_index,
            )
            link_or_copy(source_video, destination)

        task_values = episode["tasks"]
        if isinstance(task_values, np.ndarray):
            task_values = task_values.tolist()
        elif not isinstance(task_values, list):
            task_values = [str(tasks.iloc[0]["task"])]
        episode_rows.append({"episode_index": new_index, "tasks": task_values, "length": length})
        stats_rows.append({"episode_index": new_index, "stats": compute_episode_stats(frames, info)})
        mapping.append({"episode_index": new_index, "source_episode_index": source_index, "length": length})
        global_index += length

    selected_info = dict(info)
    selected_info["total_frames"] = global_index
    write_v21_info(OUTPUT, selected_info, len(selected))
    write_jsonl(OUTPUT / "meta/episodes.jsonl", episode_rows)
    write_jsonl(OUTPUT / "meta/episodes_stats.jsonl", stats_rows)
    write_jsonl(
        OUTPUT / "meta/tasks.jsonl",
        [{"task_index": int(row["task_index"]), "task": row["task"]} for _, row in tasks.iterrows()],
    )
    (OUTPUT / "meta/source_manifest.json").write_text(
        json.dumps(
            {
                "source_repo_id": "LUOSYrrrrr/so101_battery_insertion_v2",
                "source_revision": SOURCE_REVISION,
                "source_root": str(SOURCE),
                "selected_episodes": selected,
                "episode_mapping": mapping,
                "total_episodes": len(selected),
                "total_frames": global_index,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(selected)} episodes / {global_index} frames at {OUTPUT}")


if __name__ == "__main__":
    main()
