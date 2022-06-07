import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import csv

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version

from auto_model_for_task2 import AutoModelForTask2Regression, AutoModelForTask2MultiClass


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_type", type=str, required=True, help="Define task2 as either regression, multi-class, or multi-label."
    )
    parser.add_argument(
        "--predict_file", type=str, default=None, help="A csv or a json file containing the data to predict."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the testing dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    args = parser.parse_args()

    if args.predict_file is not None:
        extension = args.predict_file.split(".")[-1]
        assert extension in ["csv", "json"], "`predict_file` should be a csv or a json file."

    assert args.task_type in ["regression", "multi-class", "multi-label"], "Undefined task type. Should be \"regression\", \"multi-class\", or \"multi-label\"."

    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        logger.warning("No GPU device is detected.")

    data_files = {}

    if args.predict_file is not None:
        data_files["test"] = args.predict_file

    extension = args.predict_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    # Accitentaly sort the label and upset the order. The follow code fixes this.
    original_aspect_order = [
        "Location#Transportation",
        "Location#Downtown",
        "Location#Easy_to_find",
        "Service#Queue",
        "Service#Hospitality",
        "Service#Parking",
        "Service#Timely",
        "Price#Level",
        "Price#Cost_effective",
        "Price#Discount",
        "Ambience#Decoration",
        "Ambience#Noise",
        "Ambience#Space",
        "Ambience#Sanitary",
        "Food#Portion",
        "Food#Taste",
        "Food#Appearance",
        "Food#Recommend"
    ]
    aspect_to_original_order = {a: i for a, i in zip(original_aspect_order, range(len(original_aspect_order)))}
    sorted_aspects = sorted(original_aspect_order)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.task_type == "regression":
        model = AutoModelForTask2Regression.from_pretrained(args.model_name_or_path)
    elif args.task_type == "multi-class":
        model = AutoModelForTask2MultiClass.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError

    model.to(device)
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = ((examples["review"],))
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        result["id"] = examples["id"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = processed_datasets["test"]

    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the testing set: {test_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    test_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_batch_size
    )

    
    model.eval()
    prediction_dict = {}

    for step, batch in enumerate(test_dataloader):
        ids = batch["id"]
        del batch["id"]
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        if args.task_type == "regression":
            negative_threshold = outputs.logits < -1 / 3
            neutral_threshold = torch.logical_not(outputs.logits < -1 / 3) * (outputs.logits < 1 / 3)
            positve_threshold = torch.logical_not(outputs.logits < 1 / 3)
            predictions = torch.clone(outputs.logits)
            predictions[negative_threshold] = -1
            predictions[neutral_threshold] = 0
            predictions[positve_threshold] = 1
            predictions = predictions.int()

            for i, ps in zip(ids, predictions):
                prediction_dict[i.item()] = {}

                for j, p in enumerate(ps):
                    prediction_dict[i.item()][aspect_to_original_order[sorted_aspects[j]] + 1] = p.item()
        elif args.task_type == "multi-class":
            predictions = outputs.logits.view(-1, model.config.num_labels).argmax(dim = -1) - 1

            for i, p in enumerate(predictions):
                if (ids[i // model.config.num_aspects].item()) not in prediction_dict:
                    prediction_dict[ids[i // model.config.num_aspects].item()] = {}

                prediction_dict[ids[i // model.config.num_aspects].item()][aspect_to_original_order[sorted_aspects[i % model.config.num_aspects]] + 1] = p.item()
        else:
            raise NotImplementedError

    with open("task2_prediction.csv", "w", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id-#aspect", "sentiment"])

        for id, id_predictions in prediction_dict.items():
            for aspect, label in sorted(id_predictions.items()):
                writer.writerow([f"{id}-{aspect}", label])

if __name__ == "__main__":
    main()