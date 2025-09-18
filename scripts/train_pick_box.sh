python ./lerobot/lerobot/scripts/train.py \
  --output_dir=outputs/models/act_pick_box \
  --policy.type=act \
  --dataset.repo_id=pick_box \
  --dataset.root=outputs/datasets/pick_box \
  --wandb.enable=false \
  --steps=10000 \
  --log_freq=200 \
  --save_freq=2000 \
  --batch_size=8
