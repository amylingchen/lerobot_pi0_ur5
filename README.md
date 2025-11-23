# data preparing

[convert data to lerobot ](ur5_lerobot/README.md)

# pi0 fintune

```
cd lerobot

pip install -e .
pip install -e ".[pi]"
```


```
pip install robosuite==1.4.1 easydict bddl gym tensorflow==2.15.0 timm matplotlib seaborn

```

```
pip install numpy==1.26.4 pytest

```

### Download the model:
```
cd {this_repo}/checkpoints 
python download_model.py lerobot/pi05_base
```

### update checkpoints/pi05_base/config.json

observation.images.base_0_rgb -> observation.images.image


### Download datasets

```

from huggingface_hub import login

from huggingface_hub import snapshot_download


repo_id = 'amylingchen/Robotic_Vision_Laboratory_ur5_v30'


local_dir ="datasets/amylingchen/Robotic_Vision_Laboratory_ur5_v30"    
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False, 
    resume_download=True,
)

```

### fintune

```
nohup python lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=amylingchen/Robotic_Vision_Laboratory_ur5_v30 \
    --dataset.root=./datasets/amylingchen/Robotic_Vision_Laboratory_ur5_v30 \
    --policy.type=pi0 \
    --output_dir=./outputs/pi0_training_1 \
    --job_name=pi0_training \
    --policy.pretrained_path=./checkpoints/pi0_base \
    --policy.repo_id=pi0_ur5_rvl \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=200000 \
    --policy.scheduler_decay_steps=3000 \
    --policy.device=cuda \
    --batch_size=4 \
    > train.log 2>&1
```