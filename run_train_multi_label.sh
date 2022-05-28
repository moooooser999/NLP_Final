CUDA_DEVICES_VISIBLE=1,2,3 python3 train.py \
--project_name NLP_FINAL_macbert_base \
--model_name macbert \
--train_data_dir ./data/train.csv \
--dev_data_dir ./data/dev.csv \
--output_dir ./output \
--do_train True \
--do_validate True \
--strategy multi-label
#--model_ckpt ./output/best_model_w_adam.pt \
