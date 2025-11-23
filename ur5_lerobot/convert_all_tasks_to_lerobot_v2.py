import os
import json
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import imageio
import cv2
import random
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
vid_H = 480
vid_W = 640
CROP_H, CROP_W = 480, 480
# ============================================================
#                  基础转换函数
# ============================================================
CUR_IDX=0

taskdic = {
    0: "move the tub near the red cup",
    1: "move the tub near the blue cup",
    2: "move the tissue box farther from the orange cup",
    3: "Move the yellow box to the empty space between the two cubes"
}
reverse_taskdic = {v.lower(): k for k, v in taskdic.items()}

def load_episode(episode_dir):
    """加载一个 episode 的 npy + json 数据"""
    states = np.load(os.path.join(episode_dir, "states.npy"))
    actions = np.load(os.path.join(episode_dir, "actions.npy"))
    n_frames = min(len(states),len(actions))
    with open(os.path.join(episode_dir, "meta.json")) as f:
        meta = json.load(f)
    if len(states) > len(actions):
        states = states[:len(actions)]
    
    meta["timestamps"] = meta["timestamps"][:n_frames]
    meta['n_frames'] =n_frames
    
    return states, actions, meta


def make_dataframe(states, actions, meta, episode_index, task_index,cur_idx):
    """构建符合 LeRobot 格式的 DataFrame"""
    n_frames = min(len(states), len(actions))  
    timestamps = meta["timestamps"]
    fps = meta.get("fps", 5)    
    timestamps_sec = [t / fps for t in timestamps]
    df = pd.DataFrame({
        "observation.state": [s.tolist() for s in states],
        "action": [a.tolist() for a in actions],
        "episode_index": [episode_index] * n_frames,
        "frame_index": list(range(n_frames)),
        "timestamp": timestamps_sec,
        "index": list(range(cur_idx, cur_idx + n_frames)),
        "task_index": [task_index] * n_frames,
    })
    return df


def dataframe_to_parquet(df, save_path):
    """将 DataFrame 保存为 parquet 文件（Arrow schema）"""
    arr_state = pa.array(df["observation.state"].tolist(), type=pa.list_(pa.float32()))
    arr_action = pa.array(df["action"].tolist(), type=pa.list_(pa.float32()))

    table = pa.table({
        "observation.state": arr_state,
        "action": arr_action,
        "episode_index": pa.array(df["episode_index"], type=pa.int32()),
        "frame_index": pa.array(df["frame_index"], type=pa.int32()),
        "timestamp": pa.array(df["timestamp"], type=pa.float32()),
        "index": pa.array(df["index"], type=pa.int64()),
        "task_index": pa.array(df["task_index"], type=pa.int32()),
    })
    pq.write_table(table, save_path)
    print(f"✅ Parquet saved: {save_path}")


def center_crop(img, crop_h=CROP_H, crop_w=CROP_W):
    """裁剪图像中心 crop_h × crop_w 区域"""
    h, w = img.shape[:2]

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    return img[top:top + crop_h, left:left + crop_w]

def images_to_video(obs_dir, save_path, fps=5, drop_last=False):
    images = sorted([
        f for f in os.listdir(obs_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not images:
        raise ValueError(f"No image files found in {obs_dir}")

    if drop_last:
        images = images[:-1]

    # 使用你自己定义的 vid_H, vid_W
    H, W = CROP_H, CROP_W 

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 🔥 使用 imageio.get_writer（v2 API）
    with imageio.get_writer(save_path, fps=fps, codec="libx264") as writer:
        for img_name in images:
            img = cv2.imread(os.path.join(obs_dir, img_name))
            if img is None:
                print(f"⚠️ Skip unreadable: {img_name}")
                continue

            img = center_crop(img, H, W)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            writer.append_data(img)

    print(f"🎥 Video saved: {save_path} ({len(images)} frames, {fps} FPS)")

# ============================================================
#                  主批量处理逻辑
# ============================================================

def extract_task_index(task_name):
    """从task_name提取 task 数字"""
    match = reverse_taskdic.get(task_name.lower())
    return int(match) if match else 0


def convert_episode(episode_dir, output_base, chunk, episode_index,cur_idx=0):
    """转换单个 episode"""
    states, actions, meta = load_episode(episode_dir)
    task_index = extract_task_index(meta['instruction'])
    df = make_dataframe(states, actions, meta, episode_index, task_index,cur_idx)
    print(f" episode {episode_index:06d} action {actions.shape} frames.")
    # 输出目录结构
    log_dir = os.path.join(output_base, "data", chunk)
    obs_dir = os.path.join(episode_dir, "obs")

    vid_dir = os.path.join(output_base, "videos", chunk,'observation.images.image')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    parquet_path = os.path.join(log_dir, f"episode_{episode_index:06d}.parquet")
    video_path = os.path.join(vid_dir, f"episode_{episode_index:06d}.mp4")

    dataframe_to_parquet(df, parquet_path)
    images_to_video(obs_dir, video_path, fps=meta['fps'], drop_last=True)
    print([meta['n_frames']])
    return meta['n_frames']


def convert_all_tasks(
    root_dir=".",
    output_base="output",
    chunk="chunk_000",
):
    """
    遍历所有 task_xxx/episode_xxx 目录并批量转换。
    先随机打乱所有 episode 执行顺序，然后统一编号 episode_xxxxxx。
    
    输出结构示例：
        output/data/chunk_000/episode_000001.parquet
        output/videos/chunk_000/state/episode_000001.mp4
    """
    # -------------------------------
    # 1. 收集所有 episode 绝对路径
    # -------------------------------
    episode_paths = []

    for task in sorted(os.listdir(root_dir)):
        task_path = os.path.join(root_dir, task)
        if not os.path.isdir(task_path) or not task.startswith("task_"):
            continue

        for episode in sorted(os.listdir(task_path)):
            episode_path = os.path.join(task_path, episode)
            if not os.path.isdir(episode_path) or not episode.startswith("episode_"):
                continue

            episode_paths.append(episode_path)



    random.seed(100)      # 固定随机种子，保证可复现
    random.shuffle(episode_paths)

    print(episode_paths)


    episode_counter = 0
    cur_idx = CUR_IDX

    for episode_path in episode_paths:
        print(f"➡️  Converting episode #{episode_counter:06d}:  {episode_path}")

        # 转换
        n_frames = convert_episode(
            episode_path,
            output_base,
            chunk,
            episode_counter,
            cur_idx
        )

        # episode 累加
        episode_counter += 1
        cur_idx += n_frames

    print(f"\n✅ All done. Converted {episode_counter-1} episodes.")




# ============================================================
#                     主入口
# ============================================================

if __name__ == "__main__":
    root_dir = os.path.join(project_root, "datasets/ur5/original")
    output_base = os.path.join(project_root, "datasets/ur5/outputs")
    print(os.getcwd())
    convert_all_tasks(root_dir=root_dir, output_base=output_base)
