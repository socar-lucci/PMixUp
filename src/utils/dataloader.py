from transformers import AutoTokenizer
import torch
import pandas as pd
from transformers import BertModel, AutoTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import os
import random
import numpy as np


def get_label_dict(train_df, label_name):
    label_counts = Counter(train_df[label_name])
    label_dict = {k: n for n, k in enumerate(label_counts)}
    return label_dict


class TextDataset(Dataset):
    def __init__(self, df, label_dict, text_column, label_column, max_length):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.label_dict = label_dict
        self.device = torch.device("cuda")
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.text_column]

        label_str = self.df.iloc[idx][self.label_column]
        label = self.label_dict[label_str]
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            pad_to_max_length=True,
            max_length=self.max_length,
            truncation=True,
        )

        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "labels": torch.tensor(label).to(self.device),
        }
        return inputs