python ./lerobot/lerobot/scripts/train.py \
  --output_dir=outputs/models/act_dishwasher \
  --policy.type=act \
  --dataset.repo_id=dishwasher \
  --dataset.root=outputs/datasets/dishwasher \
  --wandb.enable=false \
  --steps=14000 \
  --log_freq=200 \
  --save_freq=2000 \
  --batch_size=16
