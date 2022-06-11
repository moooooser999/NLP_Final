from argparse import ArgumentParser, Namespace
from transformers import logging as lg
import torch
import os
import logging

from model import ForMultipleAns
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import wandb
import pandas as pd
from tqdm.auto import tqdm
from utils import (
    preprocess_for_task1,
    Task1Dataset,
    data_collator_for_train_and_dev,
    data_collator_for_test,
    macro_f1,
    precision_with_target,
    recall_with_target,
)

lg.set_verbosity_error()
DEVICE = 'cuda:1'
logger = logging.getLogger(__name__)
num_labels = 18


def to_output(pred_ls, ids_ls, output_dir):
    id_aspects = []
    predict = []
    for i, batch in enumerate(pred_ls):
        id_ls = ids_ls[i]
        batch = batch.cpu().numpy()
        for j, id_ in enumerate(id_ls):
            data = batch[j]
            data[data == 1] = id_
            predict += list(data)
            id_aspect = [f'{id_}-{k + 1}' for k in range(0, 18)]
            id_aspects += id_aspect

    df = pd.DataFrame({
        'id-#aspect': id_aspects,
        'predicted': predict
    })
    logger.info(f'Saving output to {output_dir}')
    df.to_csv(str(output_dir) + '/output.csv', index=False)


def decode(logits, threshold):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits)
    return (probs > threshold).int()


def validate(batch, tokenizer, model):
    input_text, labels = batch
    encoded_inputs = tokenizer(input_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(
            input_ids=encoded_inputs.input_ids.to(DEVICE),
            attention_mask=encoded_inputs.attention_mask.to(DEVICE),
        )
    return output, labels


def train(batch, tokenizer, model):
    input_text, labels = batch
    encoded_inputs = tokenizer(input_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    output = model(
        input_ids=encoded_inputs.input_ids.to(DEVICE),
        attention_mask=encoded_inputs.attention_mask.to(DEVICE),
        labels=labels.to(DEVICE)
    )
    loss = output.loss
    return loss


def settle_model_and_data(strategy, model_name):
    if strategy == 'multi-label':
        train_data = preprocess_for_task1(args.train_data_dir, 'train')
        dev_data = preprocess_for_task1(args.dev_data_dir, 'dev')
        if model_name == 'hfl/chinese-macbert-large':
            model = ForMultipleAns(model_name, num_labels,1024).to(DEVICE)
            return train_data, dev_data, model
        model = ForMultipleAns(model_name, num_labels).to(DEVICE)
        return train_data, dev_data, model


def settle_test_data(strategy):
    if strategy == 'multi-label':
        test_data = preprocess_for_task1(args.test_data_dir, 'test')
        return test_data


def main(args):
    # load data
    model_dic = {
        'macbert': 'hfl/chinese-macbert-base',
        'macbert_large': 'hfl/chinese-macbert-large' ,
        'chinanews': 'uer/roberta-base-finetuned-chinanews-chinese',
        'ifeng': 'uer/roberta-base-finetuned-ifeng-chinese',
        'dianping': 'uer/roberta-base-finetuned-dianping-chinese',
        'jd_full': 'uer/roberta-base-finetuned-jd-full-chinese',
    }
    wandb.init(project=args.project_name)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    config = wandb.config
    config.TRAIN_BATCH_SIZE = 8
    config.DEV_BATCH_SIZE = 8
    config.EPOCH = 10
    config.accumulate_steps = 4

    # setup output dir
    output_dir = str(args.output_dir) + '/model/' + args.model_name + '/' + args.strategy
    # init model, tokenizer and data
    model_name = model_dic[args.model_name]

    train_data, dev_data, model = settle_model_and_data(args.strategy, model_name)

    train_dataset = Task1Dataset(train_data, 'train')
    dev_dataset = Task1Dataset(dev_data, 'dev')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # optimizer = AdamW(model.parameters(), lr=1e-5)
    if args.model_ckpt is not None:
        logger.info('Loading model from checkpoint...')
        model.load_state_dict(torch.load(args.model_ckpt))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    #  Load Data
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True,
                              collate_fn=data_collator_for_train_and_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=config.DEV_BATCH_SIZE, shuffle=False,
                            collate_fn=data_collator_for_train_and_dev)

    # progress bar
    config.max_train_steps = len(train_loader)
    progress_bar_train = tqdm(range(config.EPOCH * config.max_train_steps // config.accumulate_steps))
    progress_bar_dev = tqdm(range(config.EPOCH * len(dev_loader)))

    total_training_steps = config.max_train_steps * config.EPOCH // config.accumulate_steps
    warmup_steps = total_training_steps // 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    threshold = 0.5

    best_score = 0

    for e in range(config.EPOCH):
        train_loss_global = 0.0
        val_loss_global = 0.0
        if args.do_train:
            model.train()
            train_loss = 0
            for i, batch in enumerate(train_loader):
                progress_bar_train.set_description(f'Epoch: {e}')
                loss = train(batch, tokenizer, model)

                loss /= config.accumulate_steps
                train_loss += loss
                train_loss_global += loss
                loss.backward()
                if (i + 1) % config.accumulate_steps == 0 or (i + 1) == config.max_train_steps:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress_bar_train.set_postfix(loss=train_loss)
                    progress_bar_train.update(1)
                    train_loss = 0
                if i >= config.max_train_steps:
                    logger.info('Reach Max Steps, Early Stopping')
                    break

        if args.do_validate:
            ps_val_1 = []
            rc_val_1 = []
            model.eval()
            for batch in dev_loader:
                progress_bar_dev.set_description(f'Epoch: {e}')
                output, label = validate(batch, tokenizer, model)
                loss = model.loss(output.logits.view(-1, 18).cpu(), label.float())
                val_loss_global += loss
                progress_bar_dev.set_postfix(loss=loss)
                progress_bar_dev.update(1)
                if args.strategy == 'multi-label':
                    pred = decode(output.logits, threshold)
                    ps_1 = precision_with_target(pred.cpu().numpy(), label.cpu().numpy())
                    rc_1 = recall_with_target(pred.cpu().numpy(), label.cpu().numpy())
                    ps_val_1 += ps_1
                    rc_val_1 += rc_1

            ps_score_val_1 = 0
            rc_score_val_1 = 0
            ma_f1_val_1 = 0

            if args.strategy == 'multi-label':
                ps_score_val_1 = sum(ps_val_1) / len(ps_val_1)
                rc_score_val_1 = sum(rc_val_1) / len(rc_val_1)
                ma_f1_val_1 = macro_f1(ps_score_val_1, rc_score_val_1)
            wandb.log({
                'Macro F1 Val with target': ma_f1_val_1,
                'Precision Val with target': ps_score_val_1,
                'Recall Val with target': rc_score_val_1,
            })
            if best_score < ma_f1_val_1:
                best_score = ma_f1_val_1
                logger.info(f'Best Model Saved With Macro-F1: {best_score}')
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), output_dir + f'/best_model_w_adam_dianping_{best_score:.4f}.pt')
        wandb.log({
            'Train Loss': train_loss_global / (len(train_loader) // config.accumulate_steps),
            'Val Loss': val_loss_global / len(dev_loader),
        })

    if args.do_predict:
        test_data = settle_test_data(args.strategy)
        test_dataset = Task1Dataset(test_data, 'test')
        test_dataloader = DataLoader(test_dataset, batch_size=10, collate_fn=data_collator_for_test)
        progress_bar_test = tqdm(range(len(test_dataloader)))
        ids_ls = []
        pred_ls = []
        #
        pred_logits = []
        for batch in test_dataloader:
            progress_bar_test.set_description(f'Testing')
            with torch.no_grad():
                id_ls, input_text = batch
                encoded_inputs = tokenizer(input_text, max_length=512, padding='max_length', truncation=True,
                                           return_tensors='pt')
                output = model(
                    input_ids=encoded_inputs.input_ids.to(DEVICE),
                    attention_mask=encoded_inputs.attention_mask.to(DEVICE),
                )
            ids_ls.append(id_ls)
            pred_logits.append(output.logits)
            pred_ls.append(decode(output.logits, threshold))
            progress_bar_test.update(1)
        if args.ensemble:
            pred_ensemble_logits = []
            model_name_ls = args.model_name_ensemble
            for i,model_name in enumerate(model_name_ls):
                pred_2_logits = []
                model = ForMultipleAns(model_name,18).to(DEVICE)
                logger.info('Loading ensembled model...')
                model.load_state_dict(torch.load(args.model_ckpt_ensemble[i]))
                progress_bar_test = tqdm(range(len(test_dataloader)))
                for batch in test_dataloader:
                    progress_bar_test.set_description(f'Testing_model_{i+2}')
                    with torch.no_grad():
                        id_ls, input_text = batch
                        encoded_inputs = tokenizer(input_text, max_length=512, padding='max_length', truncation=True,
                                                   return_tensors='pt')
                        output = model(
                            input_ids=encoded_inputs.input_ids.to(DEVICE),
                            attention_mask=encoded_inputs.attention_mask.to(DEVICE),
                        )
                    ids_ls.append(id_ls)
                    pred_2_logits.append(output.logits)
                    progress_bar_test.update(1)
                pred_ensemble_logits.append(pred_2_logits)
            pred_ls = []
            for i, batch in enumerate(pred_logits):
                sigmoid = torch.nn.Sigmoid()
                probs_1 = sigmoid(batch)
                for pred in pred_ensemble_logits:
                    probs_1 += sigmoid(pred[i])
                pred_ls.append((probs_1/(len(pred_ensemble_logits)+1) > threshold).int())


        to_output(pred_ls, ids_ls, args.output_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '--project_name',
        type=str,
        help='Project Name',
        required=True
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='Model Name',
        required=True
    )
    parser.add_argument(
        '--model_ckpt',
        type=Path,
        help='Dir to pretrained or finetuned model',
    )
    parser.add_argument(
        '--model_name_ensemble',
        type=Path,
        nargs='+',
        help='Name of pretrained or finetuned model, model included macbert_large, macbert, chinanews,ifeng, jd_full,dianping',
    )
    parser.add_argument(
        '--model_ckpt_ensemble',
        type=Path,
        nargs='+',
        help='Path to pretrained or finetuned model',
    )
    parser.add_argument(
        "--train_data_dir",
        type=Path,
        help='Dir to train data',
        required=True
    )
    parser.add_argument(
        "--dev_data_dir",
        type=Path,
        help='Dir to dev data',
        required=True
    )
    parser.add_argument(
        "--test_data_dir",
        type=Path,
        help='Dir to test data',
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help='Dir to output',
        required=True
    )
    parser.add_argument(
        "--do_train",
        type=bool,
        help='Set this to True to train',
    )
    parser.add_argument(
        "--do_validate",
        type=bool,
        help='Set this to True to validate',
    )
    parser.add_argument(
        "--do_predict",
        type=bool,
        help='Set this to True to predict with test data',
    )
    parser.add_argument(
        "--ensemble",
        type=bool,
        help='Set this to True to predict with two model',
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='set this to either binary or multi-label',
        required=True
    )

    args = parser.parse_args()
    if args.do_predict and args.test_data_dir is None:
        raise ValueError("Need path to test data to make prediction")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
