python3 main.py \
  --task_type multi-class \
  --model_name_or_path hfl/chinese-macbert-base \
  --train_file train.csv \
  --validation_file dev.csv \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --output_dir ./model