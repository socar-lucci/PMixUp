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
from glob import glob



def get_baseline_optimizer(model, lr = 4e-5):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def get_pmixup_optimizer(model, lr=3e-5):
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": 3e-5},
            {"params": model.module.linear.parameters(), "lr": 1e-3},
        ]
    )

    return optimizer