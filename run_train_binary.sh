python3 train.py \
--project_name NLP_FINAL_macbert_base_binary \
--model_name macbert \
--train_data_dir ./data/train.csv \
--dev_data_dir ./data/dev.csv \
--output_dir ./output \
--do_train True \
--do_validate True \
--strategy binary
