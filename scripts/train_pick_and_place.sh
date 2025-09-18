python ./lerobot/lerobot/scripts/train.py \
  --output_dir=outputs/models/act_pick_and_place \
  --policy.type=act \
  --dataset.repo_id=pick_and_place \
  --dataset.root=outputs/datasets/pick_and_place \
  --wandb.enable=false \
  --steps=10000 \
  --log_freq=200 \
  --save_freq=2000 \
  --batch_size=12