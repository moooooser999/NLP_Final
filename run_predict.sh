CUDA_DEVICES_VISIBLE=1,2,3 python3 train.py \
--project_name NLP_FINAL_PREDICT \
--model_name macbert_large \
--model_ckpt ./output/model/macbert_large/multi-label/best_model_w_adam_ifeng_0.9067.pt \
--train_data_dir ./data/train.csv \
--dev_data_dir ./data/dev.csv \
--test_data_dir ./data/test.csv \
--output_dir ./output \
--strategy multi-label \
--do_predict True\
