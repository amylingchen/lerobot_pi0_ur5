#!/usr/bin/env python3
"""
Build a LeRobot v2.1-style UR5 dataset from raw original episodes.

This version reads raw observation folders obs1/obs2 and saves them as image1/image2 in the output dataset, while explicitly fixing
state/action/image length mismatches by aligning every episode to the same
number of frames:

    n_frames = min(states, actions, timestamps if available, obs1 images, obs2 images)

Input layout:
    original/
      task_xxx/
        episode_xxx/
          states.npy
          actions.npy
          meta.json      # should contain instruction, timestamps, fps
          obs1/          # raw camera/view 1 png/jpg images, no crop/resize
          obs2/          # raw camera/view 2 png/jpg images, no crop/resize

Output layout:
    outputs/
      data/chunk_000/episode_000000.parquet
      videos/chunk_000/observation.images.image1/episode_000000.mp4
      videos/chunk_000/observation.images.image2/episode_000000.mp4
      meta/episodes.jsonl
      meta/episodes_stats.jsonl
      meta/tasks.jsonl
      meta/info.json

Example:
    python build_lerobot_v21_dataset_two_obs_aligned.py \
      --original-dir mydatasets/ur5/original \
      --output-dir mydatasets/ur5/outputs \
      --task-module ur5_lerobot.constant \
      --train-ratio 0.75 \
      --fps 30 \
      --video-fps 30 \
      --overwrite

Notes:
    Images are written to video at their original resolution. No crop/resize is applied.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Config / loading helpers
# -----------------------------

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_taskdic(task_module: str | None, task_json: str | None) -> dict[int, str]:
    if task_json:
        with open(task_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}

    if not task_module:
        raise ValueError("Provide either --task-module or --task-json")

    mod = importlib.import_module(task_module)
    if not hasattr(mod, "taskdic"):
        raise AttributeError(f"{task_module} does not define taskdic")
    return {int(k): str(v) for k, v in mod.taskdic.items()}


def natural_episode_number(path: Path) -> int:
    nums = re.findall(r"\d+", path.name)
    return int(nums[-1]) if nums else 0


def collect_episode_paths(original_dir: Path) -> list[Path]:
    if not original_dir.exists():
        raise FileNotFoundError(f"original_dir does not exist: {original_dir}")

    episode_paths: list[Path] = []
    for task_dir in sorted(original_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        for episode_dir in sorted(task_dir.iterdir(), key=natural_episode_number):
            if episode_dir.is_dir() and episode_dir.name.startswith("episode_"):
                episode_paths.append(episode_dir)
    return episode_paths


def load_episode_raw(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    states_path = episode_dir / "states.npy"
    actions_path = episode_dir / "actions.npy"
    meta_path = episode_dir / "meta.json"

    if not states_path.exists():
        raise FileNotFoundError(f"Missing states.npy: {states_path}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing actions.npy: {actions_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    states = np.load(states_path)
    actions = np.load(actions_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return states, actions, meta


def extract_task_index(instruction: str, taskdic: dict[int, str], default_task_index: int = 0) -> int:
    reverse = {v.strip().lower(): k for k, v in taskdic.items()}
    return int(reverse.get(instruction.strip().lower(), default_task_index))


# -----------------------------
# Image / video helpers
# -----------------------------


def list_images(obs_dir: Path) -> list[Path]:
    if not obs_dir.exists():
        raise FileNotFoundError(f"Observation image folder does not exist: {obs_dir}")
    images = [p for p in obs_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images, key=natural_episode_number)


def maybe_drop_last(paths: list[Path], drop_last: bool) -> list[Path]:
    if drop_last and paths:
        return paths[:-1]
    return paths


def write_images_to_video(
    image_paths: list[Path],
    save_path: Path,
    *,
    fps: int,
    max_frames: int,
) -> int:
    selected_paths = image_paths[:max_frames]
    if not selected_paths:
        raise ValueError(f"No image frames to write for {save_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with imageio.get_writer(str(save_path), fps=fps, codec="libx264") as writer:
        for image_path in selected_paths:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️ Skip unreadable image: {image_path}")
                continue

            # No crop and no resize: keep original image resolution.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            writer.append_data(img)
            written += 1

    if written != max_frames:
        raise ValueError(
            f"Video frame mismatch for {save_path}: expected {max_frames}, wrote {written}. "
            "This usually means some image files were unreadable."
        )

    return written


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


# -----------------------------
# Frame alignment
# -----------------------------


def prepare_aligned_episode(
    episode_dir: Path,
    obs_folders: list[str],
    *,
    drop_last_image: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, list[Path]], dict[str, int]]:
    """Load one episode and align states/actions/timestamps/images by min length."""
    states, actions, meta = load_episode_raw(episode_dir)

    image_paths_by_folder: dict[str, list[Path]] = {}
    image_counts: dict[str, int] = {}
    for obs_folder in obs_folders:
        paths = list_images(episode_dir / obs_folder)
        paths = maybe_drop_last(paths, drop_last_image)
        image_paths_by_folder[obs_folder] = paths
        image_counts[obs_folder] = len(paths)

    timestamps = meta.get("timestamps", None)
    has_timestamps = isinstance(timestamps, list) and len(timestamps) > 0

    candidate_lengths = {
        "states": len(states),
        "actions": len(actions),
        **{f"images/{folder}": count for folder, count in image_counts.items()},
    }
    if has_timestamps:
        candidate_lengths["timestamps"] = len(timestamps)

    n_frames = min(candidate_lengths.values())
    if n_frames <= 0:
        raise ValueError(f"Invalid aligned frame count {n_frames} in {episode_dir}: {candidate_lengths}")

    if len(set(candidate_lengths.values())) != 1:
        print(f"⚠️ Length mismatch in {episode_dir}")
        print(f"   raw lengths: {candidate_lengths}")
        print(f"   aligned n_frames = {n_frames}")

    states = states[:n_frames]
    actions = actions[:n_frames]

    if has_timestamps:
        meta["timestamps"] = timestamps[:n_frames]
    else:
        # Fallback: frame ids. make_dataframe will convert them by fps.
        meta["timestamps"] = list(range(n_frames))

    meta["n_frames"] = n_frames
    return states, actions, meta, image_paths_by_folder, candidate_lengths


# -----------------------------
# Episode conversion
# -----------------------------


def make_dataframe(
    states: np.ndarray,
    actions: np.ndarray,
    meta: dict[str, Any],
    *,
    episode_index: int,
    task_index: int,
    start_index: int,
    fps: int,
) -> pd.DataFrame:
    n_frames = int(meta["n_frames"])
    timestamps = meta.get("timestamps", list(range(n_frames)))[:n_frames]

    # Current behavior: timestamp = recorded index / fps.
    # If your timestamps are already seconds, use --timestamp-mode seconds.
    timestamp_mode = meta.get("timestamp_mode", "frame_index")
    if timestamp_mode == "seconds":
        timestamps_sec = [float(t) for t in timestamps]
    else:
        timestamps_sec = [float(t) / float(fps) for t in timestamps]

    return pd.DataFrame(
        {
            "observation.state": [s.astype(np.float32).tolist() for s in states[:n_frames]],
            "action": [a.astype(np.float32).tolist() for a in actions[:n_frames]],
            "episode_index": [episode_index] * n_frames,
            "frame_index": list(range(n_frames)),
            "timestamp": timestamps_sec,
            "index": list(range(start_index, start_index + n_frames)),
            "task_index": [task_index] * n_frames,
        }
    )


def dataframe_to_parquet(df: pd.DataFrame, save_path: Path) -> None:
    arr_state = pa.array(df["observation.state"].tolist(), type=pa.list_(pa.float32()))
    arr_action = pa.array(df["action"].tolist(), type=pa.list_(pa.float32()))

    table = pa.table(
        {
            "observation.state": arr_state,
            "action": arr_action,
            "episode_index": pa.array(df["episode_index"], type=pa.int64()),
            "frame_index": pa.array(df["frame_index"], type=pa.int64()),
            "timestamp": pa.array(df["timestamp"], type=pa.float32()),
            "index": pa.array(df["index"], type=pa.int64()),
            "task_index": pa.array(df["task_index"], type=pa.int64()),
        }
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, save_path)


def convert_all_episodes(args: argparse.Namespace, taskdic: dict[int, str]) -> tuple[int, int]:
    original_dir = Path(args.original_dir)
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data" / args.chunk
    video_root = output_dir / "videos" / args.chunk

    episode_paths = collect_episode_paths(original_dir)
    if not episode_paths:
        raise ValueError(f"No task_xxx/episode_xxx folders found under {original_dir}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(episode_paths)

    total_frames = 0
    current_global_index = args.start_index

    for episode_index, episode_dir in enumerate(episode_paths):
        states, actions, meta, image_paths_by_folder, raw_lengths = prepare_aligned_episode(
            episode_dir,
            args.obs_folders,
            drop_last_image=args.drop_last_image,
        )

        task_index = extract_task_index(meta.get("instruction", ""), taskdic, args.default_task_index)
        episode_fps = int(meta.get("fps", args.fps)) if args.use_meta_fps else args.fps
        video_fps = int(meta.get("fps", args.video_fps)) if args.use_meta_video_fps else args.video_fps

        df = make_dataframe(
            states,
            actions,
            meta,
            episode_index=episode_index,
            task_index=task_index,
            start_index=current_global_index,
            fps=episode_fps,
        )

        n_frames = len(df)
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        dataframe_to_parquet(df, parquet_path)

        written_video_frames: dict[str, int] = {}
        for obs_folder, video_key in zip(args.obs_folders, args.video_keys):
            video_path = video_root / video_key / f"episode_{episode_index:06d}.mp4"
            written_video_frames[video_key] = write_images_to_video(
                image_paths_by_folder[obs_folder],
                video_path,
                fps=video_fps,
                max_frames=n_frames,
            )

            if args.validate_video_frames:
                actual = get_video_frame_count(video_path)
                if actual != n_frames:
                    raise ValueError(
                        f"Encoded video frame mismatch for {video_path}: "
                        f"parquet={n_frames}, video={actual}"
                    )

        current_global_index += n_frames
        total_frames += n_frames

        print(
            f"✅ episode_{episode_index:06d}: task={task_index}, "
            f"frames={n_frames}, video_frames={written_video_frames}, raw_lengths={raw_lengths}"
        )

    return len(episode_paths), total_frames


# -----------------------------
# Meta generation
# -----------------------------


def sorted_parquet_files(output_dir: Path, chunk: str) -> list[Path]:
    data_dir = output_dir / "data" / chunk
    return sorted(data_dir.glob("*.parquet"), key=natural_episode_number)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def create_episodes_jsonl(output_dir: Path, chunk: str, taskdic: dict[int, str]) -> None:
    rows = []
    for parquet_path in sorted_parquet_files(output_dir, chunk):
        df = pd.read_parquet(parquet_path)
        episode_index = int(df["episode_index"].iloc[0])
        task_index = int(df["task_index"].iloc[0])
        rows.append(
            {
                "episode_index": episode_index,
                "tasks": [taskdic.get(task_index, f"Unknown task {task_index}")],
                "length": int(len(df)),
            }
        )
    write_jsonl(output_dir / "meta" / "episodes.jsonl", rows)
    print(f"✅ Saved episodes.jsonl ({len(rows)} episodes)")


def create_tasks_jsonl(output_dir: Path, taskdic: dict[int, str]) -> None:
    rows = [{"task_index": int(k), "task": v} for k, v in sorted(taskdic.items())]
    write_jsonl(output_dir / "meta" / "tasks.jsonl", rows)
    print(f"✅ Saved tasks.jsonl ({len(rows)} tasks)")


def estimate_num_samples(dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def get_feature_stats(array: np.ndarray, axis: tuple | int | None, keepdims: bool) -> dict[str, Any]:
    array = np.asarray(array)
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": np.array([len(array)]).tolist(),
    }


def sample_video_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")

    num_samples = estimate_num_samples(frame_count)
    indices = set(np.round(np.linspace(0, frame_count - 1, num_samples)).astype(int).tolist())

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)
            chw = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)
            frames.append(chw)
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No sampled frames from {video_path}")
    return np.stack(frames, axis=0)


def create_episodes_stats_jsonl(output_dir: Path, chunk: str, video_keys: list[str]) -> None:
    rows = []
    for parquet_path in sorted_parquet_files(output_dir, chunk):
        df = pd.read_parquet(parquet_path)
        episode_index = int(df["episode_index"].iloc[0])

        row: dict[str, Any] = {"episode_index": episode_index, "stats": {}}

        for video_key in video_keys:
            video_path = output_dir / "videos" / chunk / video_key / parquet_path.name.replace(".parquet", ".mp4")
            frames = sample_video_frames(video_path)
            img_stats = {
                k: (v if k == "count" else np.squeeze(v / 255.0, axis=0)).tolist()
                for k, v in {
                    "min": np.min(frames, axis=(0, 2, 3), keepdims=True),
                    "max": np.max(frames, axis=(0, 2, 3), keepdims=True),
                    "mean": np.mean(frames, axis=(0, 2, 3), keepdims=True),
                    "std": np.std(frames, axis=(0, 2, 3), keepdims=True),
                    "count": np.array([len(frames)]),
                }.items()
            }
            row["stats"][video_key] = img_stats

        if video_keys:
            row["stats"]["observation.image"] = row["stats"][video_keys[0]]

        row["stats"]["observation.state"] = get_feature_stats(
            np.vstack(df["observation.state"].to_numpy()), axis=0, keepdims=False
        )
        row["stats"]["action"] = get_feature_stats(np.vstack(df["action"].to_numpy()), axis=0, keepdims=False)
        for key in ["episode_index", "frame_index", "timestamp", "index", "task_index"]:
            row["stats"][key] = get_feature_stats(df[key].to_numpy(), axis=0, keepdims=True)

        rows.append(row)

    write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", rows)
    print(f"✅ Saved episodes_stats.jsonl ({len(rows)} episodes)")


def get_dir_size_mb(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return int(round(total / (1024 * 1024)))


def get_first_video_shape(output_dir: Path, chunk: str, video_key: str) -> tuple[int, int]:
    video_dir = output_dir / "videos" / chunk / video_key
    first_video = next(iter(sorted(video_dir.glob("*.mp4"), key=natural_episode_number)), None)
    if first_video is None:
        raise ValueError(f"No videos found under {video_dir}")

    cap = cv2.VideoCapture(str(first_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot infer video size from {first_video}")
    return height, width


def create_info_json(args: argparse.Namespace, taskdic: dict[int, str], total_episodes: int, total_frames: int) -> None:
    output_dir = Path(args.output_dir)
    video_shapes = {video_key: get_first_video_shape(output_dir, args.chunk, video_key) for video_key in args.video_keys}
    train_end = int(total_episodes * args.train_ratio)

    info = {
        "codebase_version": "v2.1",
        "trossen_subversion": "v1.0",
        "robot_type": args.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(taskdic),
        "total_chunks": (total_episodes + args.chunks_size - 1) // args.chunks_size,
        "total_videos": total_episodes * len(args.video_keys),
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "splits": {
            "train": f"0:{train_end}",
            "val": f"{train_end}:{total_episodes}",
        },
        "data_path": f"data/{{episode_chunk:03d}}/episode_{{episode_index:06d}}.parquet",
        "video_path": f"videos/{{episode_chunk:03d}}/{{video_key}}/episode_{{episode_index:06d}}.mp4",
        "features": {
            **{
                video_key: {
                    "dtype": "video",
                    "shape": [video_shapes[video_key][0], video_shapes[video_key][1], 3],
                    "names": ["height", "width", "rgb"],
                    "info": {
                        "video.height": video_shapes[video_key][0],
                        "video.width": video_shapes[video_key][1],
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "video.fps": args.video_fps,
                        "video.channels": 3,
                        "has_audio": False,
                    },
                }
                for video_key in args.video_keys
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [args.state_dim],
                "names": {"motors": args.state_names},
            },
            "action": {
                "dtype": "float32",
                "shape": [args.action_dim],
                "names": {"motors": args.action_names},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "data_files_size_in_mb": get_dir_size_mb(output_dir / "data"),
        "video_files_size_in_mb": get_dir_size_mb(output_dir / "videos"),
    }

    out_path = output_dir / "meta" / "info.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    print(f"✅ Saved info.json to {out_path}")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned UR5 LeRobot v2.1 dataset with two observations")

    parser.add_argument("--original-dir", default="mydatasets/ur5/original")
    parser.add_argument("--output-dir", default="mydatasets/ur5/outputs")
    parser.add_argument("--task-module", default="ur5_lerobot.constant", help="Python module containing taskdic")
    parser.add_argument("--task-json", default=None, help="Optional JSON mapping task_index to instruction")

    parser.add_argument("--chunk", default="chunk_000")
    # Input folders are obs1/obs2 by default.
    # Output video feature keys are image1/image2, so the saved dataset uses image names instead of raw obs names.
    parser.add_argument("--video-keys", nargs="+", default=["observation.images.image1", "observation.images.image2"])
    parser.add_argument("--obs-folders", nargs="+", default=["obs1", "obs2"])
    parser.add_argument("--robot-type", default="ur5")

    parser.add_argument("--fps", type=int, default=30, help="Dataset/control FPS used in timestamps and metadata")
    parser.add_argument("--video-fps", type=int, default=20, help="Video FPS written to mp4 and info.json")
    parser.add_argument("--use-meta-fps", action="store_true", help="Use meta.json fps for timestamps instead of --fps")
    parser.add_argument("--use-meta-video-fps", action="store_true", help="Use meta.json fps for video instead of --video-fps")

    parser.add_argument(
        "--drop-last-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop the last image in each obs folder before length alignment. Use this only if your logger always saves one extra image.",
    )
    parser.add_argument(
        "--validate-video-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check encoded mp4 frame count equals parquet frame count.",
    )

    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-names", nargs="+", default=["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
    parser.add_argument("--action-names", nargs="+", default=["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"])

    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--default-task-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Delete output-dir before building")
    parser.add_argument("--skip-stats", action="store_true", help="Skip episodes_stats.jsonl to save time")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.obs_folders) != len(args.video_keys):
        raise ValueError(f"obs_folders length {len(args.obs_folders)} != video_keys length {len(args.video_keys)}")
    if len(args.state_names) != args.state_dim:
        raise ValueError(f"state_names length {len(args.state_names)} != state_dim {args.state_dim}")
    if len(args.action_names) != args.action_dim:
        raise ValueError(f"action_names length {len(args.action_names)} != action_dim {args.action_dim}")
    if not 0 < args.train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {args.train_ratio}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    taskdic = load_taskdic(args.task_module, args.task_json)

    print("🚀 Building aligned LeRobot v2.1 dataset")
    print(f"original_dir = {args.original_dir}")
    print(f"output_dir   = {args.output_dir}")
    print(f"obs_folders  = {args.obs_folders}")
    print(f"video_keys   = {args.video_keys}")
    print(f"drop_last_image = {args.drop_last_image}")

    total_episodes, total_frames = convert_all_episodes(args, taskdic)
    create_episodes_jsonl(output_dir, args.chunk, taskdic)
    create_tasks_jsonl(output_dir, taskdic)
    if not args.skip_stats:
        create_episodes_stats_jsonl(output_dir, args.chunk, args.video_keys)
    create_info_json(args, taskdic, total_episodes, total_frames)

    print("\n🎉 Done")
    print(f"episodes = {total_episodes}")
    print(f"frames   = {total_frames}")
    print(f"output   = {output_dir}")


if __name__ == "__main__":
    main()
