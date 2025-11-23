import os
import pandas as pd
import pyarrow.parquet as pq
import json
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
# 数据目录
mypath = os.path.join(project_root, "datasets/ur5/outputs/data", "chunk_000")

taskdic = {
    0: "move the tub near the red cup",
    1: "move the tub near the blue cup",
    2: "move the tissue box farther from the orange cup",
    3: "Move the yellow box to the empty space between the two cubes"
}


# 只取 .parquet 文件并按数字排序
onlyfiles = sorted(
    [f for f in os.listdir(mypath) if f.endswith(".parquet")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

jsonl_data = []

for file in onlyfiles:
    path = os.path.join(mypath, file)
    df = pd.read_parquet(path)
    first_idx = df["index"].iat[0]
    last_idx = df["index"].iat[-1]
    episode_index = int(df["episode_index"].iat[0])
    task_index = int(df["task_index"].iat[0])
    task_desc = taskdic.get(task_index, f"Unknown task {task_index}")
    length = len(df)

    print(f"file: {file} | first index: {first_idx} | last index: {last_idx} | length: {length}")

    episode_dic = {
        "episode_index": episode_index,
        "tasks": [task_desc],
        "length": length
    }

    jsonl_data.append(episode_dic)

# 保存 JSONL 文件
out_path = os.path.join(project_root,"datasets/ur5/outputs/meta", "episodes.jsonl")
with open(out_path, "w") as f:
    for l in jsonl_data:
        f.write(json.dumps(l) + "\n")

print(f"\n✅ Saved {len(jsonl_data)} episodes to {out_path}")
