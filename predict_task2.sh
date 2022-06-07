python3 predict.py \
  --task_type multi-class \
  --model_name_or_path model_multi_class_base/epoch_4 \
  --predict_file test.csv \
  --max_length 512 \
  --per_device_batch_size 8 \