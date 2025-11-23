###  上传文件到服务器

scp -i ~/.ssh/{ssh公钥} -r  "/d:{本地文件地址}"  lxc4866@ssh.rvlab.app:{project_path}/datasets/ur5/original/


### 转换文件为lerobot v21格式

1. 运行 `convert_all_tasks_to_lerobot_v2.py` 生成video和data

2. 'create_episodes_jsonl.py' 生成episodes.jsonl

3. 'create_episodes_stats_jsonl.py' 生成 episodes_stats.jsonl

4. 修改 info.json

### 上传 数据到huggingface

1. 创建v21数据库`Robotic_Vision_Laboratory_ur5_v21`并绑定


```
from huggingface_hub import snapshot_download

# 设置你的数据集 repo ID
repo_id = 'amylingchen/Robotic_Vision_Laboratory_ur5_v21'

# 设置本地缓存路径
local_dir ="../datasets/amylingchen/Robotic_Vision_Laboratory_ur5_v21"    
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 复制而不是软链，适合在远程服务器使用
    resume_download=True,
    # ignore_patterns=["*.mp4"]  # 如果不想下载视频可加上，或者删掉这个行全部下载
)
```

2. 将 `dataset/ur5/output`的数据复制到`/datasets/amylingchen/Robotic_Vision_Laboratory_ur5_v21`

3. 上传到huggingface

```
from huggingface_hub import upload_folder

upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="dataset",

)

````
4. 给数据添加v21 版本

```
from huggingface_hub import HfApi
api = HfApi()

api.create_tag(
    repo_id="amylingchen/Robotic_Vision_Laboratory_ur5_v21",
    tag="v2.1",
    repo_type="dataset"
)
```


5. 将v21 转为v30
```
python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id amylingchen/Robotic_Vision_Laboratory_ur5_v21 \
    --root datasets/amylingchen/Robotic_Vision_Laboratory_ur5_v21 \
    --push-to-hub False \
    --force-conversion
```

6. 上传v30数据到 Hugging Face Hub 

```
repo_id = 'amylingchen/Robotic_Vision_Laboratory_ur5_v30'

local_dir ="../datasets/amylingchen/Robotic_Vision_Laboratory_ur5_v30"  

upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="dataset",

)
```