CUDA_DEVICES_VISIBLE=1,2,3 python3 train.py \
--project_name NLP_FINAL_PREDICT \
--model_ckpt ./output/best_model_w_adam_0.835668094846977.pt  \
--train_data_dir ./data/train.csv \
--dev_data_dir ./data/dev.csv \
--test_data_dir ./data/test.csv \
--output_dir ./output \
--do_predict True\
