export CUDA_VISIBLE_DEVICES=1

model_name=Informerfuzzy or Crossformerfuzzy or FuzzyiTransformer or PatchTSTfuzzy or Transformerfuzzy

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWAT \
  --model $model_name$ \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3