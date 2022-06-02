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
from transformers import AutoTokenizer
from utils.utils import seed_everything, get_pmixup_optimizer
from utils.trainer import run_pmixup
from models.tmix import *


def make_mini_sample(dataset,sample_size):
    train_df = pd.read_csv(f"../dataset/{dataset}/train.csv")
    sample_texts, sample_labels = [], []
    for key, values in tqdm(Counter(train_df['label']).items()):
        tmp = [train_df.iloc[i]['text'] for i in range(len(train_df)) if train_df.iloc[i]['label'] == key]
        random_texts = random.sample(tmp, sample_size)
        sample_texts += random_texts
        labs = [key for _ in range(sample_size)]
        sample_labels += labs
    new_df = pd.DataFrame({"text": sample_texts, "label": sample_labels})
    new_df.to_csv(f"../dataset/{dataset}/train_{sample_size}.csv")
    return new_df



def main():
    ## IMP, POS가 없다면 IMP, POS AUG 만들기
    seed_everything()
    datasets = ["stackoverflow"]
    sample_sizes = [10, 250, 2000]
    for dataset in datasets:
        mini_df = make_mini_sample(dataset, sample_size = 10)



    train_df = pd.read_csv("../dataset/stackoverflow/train_noun_aug.csv")
    dataset = "stackoverflow"
    feature = "noun"
    run_pmixup(train_df, dataset, feature)



if __name__ == "__main__":
    main()
