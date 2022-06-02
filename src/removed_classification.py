import argparse
import torch
import pandas as pd
from transformers import BertModel, AutoTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
import os
import random
import numpy as np
from glob import glob
from utils.dataloader import TextDataset, get_label_dict
from utils.utils import get_baseline_optimizer
from utils.trainer import run_baseline
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download("omw-1.4")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')



def remove_token(tagged, pos):
    pos_dict = {"noun": "N", "verb": "V", "adj": "J"}
    if (len(" ".join(["" if token[1].startswith(pos_dict[pos]) else token[0] for token in tagged]))== 0):
        removed_noun = " ".join(["[MASK]" if token[1].startswith("N") else token[0] for token in tagged])
    else:
        removed_noun = " ".join(["" if token[1].startswith("N") else token[0] for token in tagged])


def save_pos_file(train_df, pos, pos_removed, text_col, label_col, datapath):
    out_dir = datapath + f"/POS/{pos}_removed"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    removed_train = pd.DataFrame(
        {text_col: pos_removed, label_col: train_df[label_col]}
    )
    removed_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)


def make_pos_files(train_df, datapath, text_col="text", label_col="label"):
    pos_list = ["noun", "verb", "adj"]
    removed_nouns, removed_verbs, removed_adjs = [], [], []
    for i in tqdm(range(len(train_df))):
        text = train_df.iloc[i][text_col]
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        if (len(" ".join(["" if token[1].startswith("N") else token[0] for token in tagged]))== 0):
            removed_noun = " ".join(["[MASK]" if token[1].startswith("N") else token[0] for token in tagged])
        else:
            removed_noun = " ".join(["" if token[1].startswith("N") else token[0] for token in tagged])
        if (len(" ".join(["" if token[1].startswith("V") else token[0] for token in tagged])) == 0):
            removed_verb = " ".join(["[MASK]" if token[1].startswith("V") else token[0] for token in tagged])
        else:
            removed_verb = " ".join(["" if token[1].startswith("V") else token[0] for token in tagged])

        if (len(" ".join(["" if token[1].startswith("J") else token[0] for token in tagged]))== 0):
            removed_adj = " ".join(["[MASK]" if token[1].startswith("J") else token[0] for token in tagged])
        else:removed_adj = " ".join(["" if token[1].startswith("J") else token[0] for token in tagged])

        removed_nouns.append(removed_noun)
        removed_verbs.append(removed_verb)
        removed_adjs.append(removed_adj)
    save_pos_file(train_df,"noun", removed_nouns, text_col, label_col, datapath)
    save_pos_file(train_df,"verb", removed_verbs, text_col, label_col, datapath)
    save_pos_file(train_df,"adj", removed_adjs, text_col, label_col, datapath)



def remove_important(dataset):
    dataframe = pd.read_csv(f"../dataset/{dataset}/train.csv")
    imp_list = pd.read_csv(f"../dataset/{dataset}/imp_removed.csv")
    for ind in tqdm(range(len(dataframe))):
        text = dataframe.iloc[i]




def main(args):
    print(f'----- Start Training -----')
    for dataset in args.datasets:
        train_df = pd.read_csv(f"../dataset/{dataset}/train.csv")
        print("-- Making POS Files! --")
        make_pos_files(train_df, f"../dataset/{dataset}")
        for pos in args.pos:
            removed = pd.read_csv(f"../dataset/{dataset}/POS/{pos}_removed/train.csv")
            run_baseline(args, removed, dataset, feature= pos, condition = 'removed')

    print(f'----- Start Evaluation -----')
    if args.eval == True:
        for dataset in args.datasets:
            for pos in args.pos:
                model = f'../model_weights/{dataset}/model_{pos}_removed.pt'
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default = ["agnews","dbpedia","stackoverflow","banking","r8","ohsumed","amazon","yelp","imdb"])
    parser.add_argument("--model_name", default = "bert-base-uncased")
    parser.add_argument("--num_epochs", default = 1)
    parser.add_argument("--lr", default = 4e-5)
    parser.add_argument('--batch_size',  default = 128)
    parser.add_argument('--max_length',  default = 100)
    parser.add_argument('--pos', nargs= "+", default = ["noun", "verb", "adj"])
    parser.add_argument('--eval', default = True)
    args = parser.parse_args()
    main(args)