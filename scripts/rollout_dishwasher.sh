python ./imitation_learning_lerobot/scripts/rollout.py \
  --policy.path=outputs/models/act_dishwasher/checkpoints/014000/pretrained_model \
  --env.type=dishwasher \
  --policy.device=cuda \
  --policy.use_amp=false