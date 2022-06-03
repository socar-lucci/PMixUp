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
import argparse
from pos_in_important import *
from pos_augmentation import *

def make_mini_sample(dataframe,sample_size):
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



def main(args):
    seed_everything()
    datasets = args.datasets
    sample_sizes = args.sample_per_class

    for dataset in datasets:
        for sample_size in sample_sizes:
            print(f'----- Making Dataset for {sample_size} Samples Per Class -------')
            mini_df = make_mini_sample(dataset, sample_size)
            out_dir1 = f'../dataset/{dataset}/imp_removed_{sample_size}.csv'
            out_dir2 = f'../dataset/{dataset}/imp_list_{sample_size}.csv'
            imp_removed = make_important_tokens(mini_df, dataset, out_dir1, out_dir2)
            imp_tokens = pd.read_csv(out_dir2)['tokens'].tolist()

            auged_df = important_augmentation(mini_df, imp_tokens)
            run_pmixup(args, auged_df, dataset, sample_size, feature = "imp")
            for pos in args.pos:
                pos_aug_df = pos_augmentation(mini_df,pos)
                run_pmixup(args, pos_aug_df, dataset, sample_size, feature = pos)


    ## Evaluation ##

    if args.eval == True:
        for dataset in args.datasets:
            for pos in args.pos:
                for sample_size in args.sample_per_class:
                    model = torch.load(f'../model_weights/{dataset}/model_{pos}_{sample_size}.pt')
                    val_dataframe = pd.read_csv(f"../dataset/{dataset}/test.csv")
                    label_dict = get_label_dict(val_dataframe, 'label')
                    val_dataset = TextDataset(val_dataframe, label_dict, "text", "label",args.max_length)
                    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
                    _, val_acc, _ = pmixup_evaluate(model, val_dataloader)
                    print(val_acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default = ["agnews","dbpedia","stackoverflow","banking","r8","ohsumed","amazon","yelp","imdb"])
    parser.add_argument("--lr", default = 3e-5)
    parser.add_argument('--batch_size',  default = 64)
    parser.add_argument('--max_length',  default = 100)
    parser.add_argument('--sample_per_class', nargs = "+", default = [10,200,600])
    parser.add_argument('--pos', nargs= "+", default = ["noun", "verb", "adj"])
    parser.add_argument('--model_name', default = "bert-base-uncased")
    parser.add_argument('--num_epochs', default = 20)
    parser.add_argument('--eval', default = True)
    args = parser.parse_args()
    main(args)
