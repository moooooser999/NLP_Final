CUDA_DEVICES_VISIBLE=1,2,3 python3 train.py \
--project_name NLP_FINAL_roberta_base_finetuned \
--model_name roberta \
--train_data_dir ./data/train.csv \
--dev_data_dir ./data/dev.csv \
--output_dir ./output \
--do_train True \
--do_validate True \
--strategy multi-label \
#--model_ckpt ./output/model/roberta/multi-label/best_model_w_adam_jd_full_867.3656782874995.pt
