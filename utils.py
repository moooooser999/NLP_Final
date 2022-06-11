import pandas as pd
import torch

from typing import List, Dict

from sklearn.metrics import precision_score, recall_score

aspect_ls = [
    'Location#Transportation',
    'Location#Downtown',
    'Location#Easy_to_find',
    'Service#Queue',
    'Service#Hospitality',
    'Service#Parking',
    'Service#Timely',
    'Price#Level',
    'Price#Cost_effective',
    'Price#Discount',
    'Ambience#Decoration',
    'Ambience#Noise',
    'Ambience#Space',
    'Ambience#Sanitary',
    'Food#Portion',
    'Food#Taste',
    'Food#Appearance',
    'Food#Recommend'
]


class Task1Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict], split):
        self.data = data
        self.split = split

        if self.split != 'test':
            self.labels = [d['labels'] for d in data]

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    def __len__(self):
        if self.split != 'test':
            return len(self.labels)
        return len(self.data)


def data_collator_for_train_and_dev(batch):
    inputs_text_ls = [data['inputs'] for data in batch]
    label_ts = torch.tensor([data['labels'] for data in batch])
    return inputs_text_ls, label_ts


def data_collator_for_test(batch):
    id_ls = [data['id'] for data in batch]
    inputs_text_ls = [data['inputs'] for data in batch]
    return id_ls, inputs_text_ls


def preprocess_for_task1(data_path: str, split: str) -> List[Dict]:
    data_list = []
    if split == "train" or split == "dev":
        data = pd.read_csv(data_path)
        for i in range(len(data)):
            data_point = data.iloc[i]
            data_dic = {
                'id': data_point[0],
                'inputs': data_point[1],
                'labels': []
            }
            for j in range(2, len(data_point)):
                if data_point[j] == -2:
                    data_dic['labels'].append(0)
                else:
                    data_dic['labels'].append(1)
            data_list.append(data_dic)
    else:
        data = pd.read_csv(data_path)
        for i in range(len(data)):
            data_point = data.iloc[i]
            data_dic = {
                'id': data_point[0],
                'inputs': data_point[1],
            }
            data_list.append(data_dic)
    return data_list


def preprocess_for_task1_binary(data_path: str, split: str) -> List[Dict]:
    data_list = []
    if split == "train" or split == "dev":
        data = pd.read_csv(data_path)
        for i in range(len(data)):
            data_point = data.iloc[i]
            for j in range(2, len(data_point)):
                data_dic = {
                    'id': data_point[0],
                    'inputs': data_point[1] + ' [aspect] ' + aspect_ls[j - 2]
                }
                if data_point[j] == -2:
                    data_dic['labels'] = 0
                else:
                    data_dic['labels'] = 1
                data_list.append(data_dic)
    else:
        data = pd.read_csv(data_path)
        for i in range(len(data)):
            data_point = data.iloc[i]
            for j in range(18):
                data_dic = {
                    'id': data_point[0],
                    'inputs': data_point[1] + ' [aspect] ' + aspect_ls[j],
                }
                data_list.append(data_dic)
    return data_list


def precision_with_category(y_pred, y_label):
    return list(precision_score(y_label, y_pred, average=None))


def recall_with_category(y_pred, y_label):
    return list(recall_score(y_label, y_pred, average=None))


def precision_with_target(y_pred, y_label):
    return list(precision_score(y_label.T, y_pred.T, average=None))


def recall_with_target(y_pred, y_label):
    return list(recall_score(y_label.T, y_pred.T, average=None))


def macro_f1(precision, recall):
    return 2 * precision * recall / (precision + recall)
