export CUDA_VISIBLE_DEVICES=2

model_name=Informerfuzzy or Crossformerfuzzy or FuzzyiTransformer or PatchTSTfuzzy or Transformerfuzzy

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model $model_name$ \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 10