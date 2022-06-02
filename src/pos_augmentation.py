import argparse
import torch
import pandas as pd
from transformers import BertModel, AutoTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter
from nltk.corpus import wordnet
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
import os
import random
import numpy as np
from glob import glob
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from utils.dataloader import *
from utils.utils import *
from utils.trainer import run_baseline
nltk.download("omw-1.4")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')



def get_label_dict(train_df, label_name):
    label_counts = Counter(train_df[label_name])
    label_dict = {k:n for n,k in enumerate(label_counts)}
    return label_dict


def words_with_synonyms(text, percentage_swap):
    word_indices = []
    nums_to_swap = int(len(text.split())*percentage_swap) + 1
    for ind,word in enumerate(text.split()):
        word_count = 0
        for syn in wordnet.synsets(word, lang = "eng"):
            words = [syn_word for syn_word in syn.lemma_names(lang = "eng") if syn_word.lower() != word.lower()]
            word_count += 1
        if word_count != 0:
            word_indices.append(ind)
    return word_indices
        

def synonym_count(word):
    nums_to_swap = 2
    cnt = 0
    all_words = []
    for syn in wordnet.synsets(word, lang = "eng"):
        words = [syn_word for syn_word in syn.lemma_names(lang= "eng") if syn_word.lower() != word.lower()]
        all_words += words
    return list(set(all_words)), len(all_words)


def important_augmentation(train_df, imp_tokens):
    nope = []
    yess = []
    syn_dict = {}
    for word in imp_tokens:
        if word not in syn_dict:
            syn_dict[word] = []
        syns, cnt = synonym_count(str(word))
        if cnt == 0:
            if "#" not in word and word.isalpha():
                nope.append(word)
        else:
            yess.append(word)
            syn_dict[word] += syns
    ### Important Token Replacement ###
    auged_text, auged_label = [], []
    shuffled_text, shuffled_label = [], []
    for ind in tqdm(range(len(train_df))):
        new_text, new_label = [], []
        
        data = train_df.iloc[ind]
        text, label = str(data['text']).lower(), data['label']
        splitted_text = text.split()
        imp_tok = imp_tokens[ind]
        if imp_tok in yess:
            synss = list(set(syn_dict[imp_tok]))
            for syn in synss[:2]:
                tmp_text = text
                nt = tmp_text.replace(str(imp_tok), str(syn))
                new_text.append(nt)
                new_label.append(label)
        if len(new_text) < 2:
            for charswap in range(len(new_text), 2):
                tmp_text = text
                new_word = "".join(random.sample(list(imp_tok),len(list(imp_tok))))
                nt = tmp_text.replace(imp_tok, new_word)
                new_text.append(nt)
                new_label.append(label)
        auged_text += new_text
        auged_label += new_label
    auged_df = pd.DataFrame({"text": auged_text, "label": auged_label})
    auged_df.to_csv("../dataset/stackoverflow/imp_auged_train.csv", index= False)

    return auged_df
            



def pos_augmentation(train_df, pos):
    pos_dict = {"verb": "V", "noun": "N", "adj": "J"}
    ### POS Token Replacement ###
    n = 5
    ni_aug, ni_lab = [], []
    for ind in tqdm(range(len(train_df))):
        new_text, new_label = [], []
            
        data = train_df.iloc[ind]
        text, label = data['text'].lower(), data['label']
        text = " " + text
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        pos_filt = pos_dict[pos]
        nouns = [[i,word[0]] for i,word in enumerate(tagged) if word[1].startswith(pos_filt)]
        possible_swaps = []
        splitted = text.split()
        for ind,swap_word in nouns:
            to_change, length = synonym_count(swap_word)
            to_change = list(set([i for i in to_change if "_" not in i]))
            if len(to_change) >= n:
                possible_swaps.append([ind,swap_word, to_change])
            else:
                tange = []
                for _ in range(n):
                    sw = "".join(random.sample(list(swap_word),len(list(swap_word))))
                    tange.append(sw)
                        
                possible_swaps.append([ind,swap_word, tange])
        tmp_text = text

        for swaps in possible_swaps[:n]:
            inde, s, c = swaps
            for swap_cnt in range(1):
                tmp_text_ = tmp_text.replace(f" {s}", f" {c[swap_cnt]}")
                new_text.append(tmp_text_)
                new_label.append(label)

        ni_aug += new_text
        ni_lab += new_label
    pos_auged = pd.DataFrame({"text": ni_aug, "label" : ni_lab})
            
    pos_auged = pd.DataFrame({"text": ni_aug, "label" : ni_lab})
    pos_auged.to_csv(f"../dataset/stackoverflow/train_{pos}_aug.csv", index=False)
    return pos_auged


def run_imp_aug(args,dataset):
    dataframe = pd.read_csv(f"../dataset/{dataset}/train.csv")
    imp_list = pd.read_csv(f"../dataset/{dataset}/imp_list.csv")['tokens'].tolist()
    auged_df = important_augmentation(dataframe, imp_list)
    run_baseline(args, auged_df, dataset, feature = "imp", condition = "auged")


def run_pos_aug(args,dataset,pos):
    dataframe = pd.read_csv(f"../dataset/{dataset}/train.csv")
    pos_auged = pos_augmentation(dataframe,pos)
    run_baseline(args, pos_auged, dataset, feature = pos, condition = "auged")
    


def main(args):
    seed_everything()
    for dataset in args.datasets:
        run_imp_aug(args, dataset)
        for pos in args.pos:
            run_pos_aug(args, dataset,pos)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default = ["agnews","dbpedia","stackoverflow","banking","r8","ohsumed","amazon","yelp","imdb"])
    parser.add_argument("--model_name", default = "bert-base-uncased")
    parser.add_argument("--num_epochs", default = 1)
    parser.add_argument("--lr", default = 4e-5)
    parser.add_argument('--batch_size',  default = 128)
    parser.add_argument('--max_length',  default = 100)
    parser.add_argument('--pos', nargs= "+", default = ["noun", "verb", "adj"])
    args = parser.parse_args()
    main(args)