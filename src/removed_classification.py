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



def run_baseline(train_df, dataset_name, pos,text_column='text', label_column='label', model_name='bert-base-uncased', num_epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else cpu )
    lr = 4e-5
    max_length = 100
    batch_size = 124

    label_dict = get_label_dict(train_df, label_column)
    train_dataset = TextDataset(train_df, label_dict, text_column, label_column, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels = len(label_dict))
    model = torch.nn.DataParallel(model).to(device)
    optimizer = get_baseline_optimizer(model, lr)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch['input_ids'] = batch['input_ids'].squeeze(1)
            with torch.cuda.amp.autocast():
                output = model(**batch)
                loss = output['loss']        
            epoch_loss += loss.mean().item()
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
    
    if not os.path.exists(f'../model_weights/{dataset_name}'):
        os.mkdir(f"../model_weights/{dataset_name}")
    torch.save(model, f'../model_weights/{dataset_name}/model_{pos}_removed.pt')



def remove_important(dataset):
    dataframe = pd.read_csv(f"../dataset/{dataset}/train.csv")
    imp_list = pd.read_csv(f"../dataset/{dataset}/imp_removed.csv")
    for ind in tqdm(range(len(dataframe))):
        text = dataframe.iloc[i]




def main():
    dataset = "stackoverflow"
    train_df = pd.read_csv(f"../dataset/{dataset}/train.csv")
    print("Making POS Files!")
    make_pos_files(train_df, f"../dataset/{dataset}")
    for pos in ["verb", "noun", "adj"]:
        removed = pd.read_csv(f"../dataset/{dataset}/POS/{pos}_removed/train.csv")
        run_baseline(removed, dataset,pos)
    
    print("Training Important Removed!")
    dataframe = pd.read_csv(f'../dataset/{dataset}/imp_removed.csv')
    run_baseline(dataframe, dataset, 'imp')

if __name__ == "__main__":
    main()