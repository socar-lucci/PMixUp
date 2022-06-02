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
from utils.utils import get_baseline_optimizer, seed_everything
from utils.trainer import run_baseline
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download("omw-1.4")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')



def make_important_tokens(train_df, project_name, out_dir1, out_dir2, text_column= "text", label_column="label"):
    all_sig = []
    important_tokens = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    label_dict = get_label_dict(train_df, label_column)
    max_length_dict = {"dbpedia": 128, "oh": 500, "agnews": 400}
    if project_name in max_length_dict:
        max_length = max_length_dict[project_name]
    else:
        max_length = 100
    
    model_path = f"../model_weights/model_{project_name}_baseline.pt"
    model = torch.load(model_path)
    print("Model Loaded!")

    device = torch.device("cuda" if torch.cuda.is_available() else cpu )
    model = model.to(device)
    for ex in tqdm(range(len(train_df))):
        text = train_df[text_column][ex]
        label = train_df[label_column][ex]
        label_ind = label_dict[label]
        tokenized = ["[CLS]"] + tokenizer.tokenize(text)[:max_length] + ["[EOS]"]
        significance = []
        for i in range(1, len(tokenized) - 1):
            ins = tokenized[:i] + tokenized[i + 1 :]
            with torch.no_grad():
                model.eval()
                input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(ins)).unsqueeze(0)
                token_type_ids = torch.tensor([0 for _ in range(len(ins))]).unsqueeze(0)
                attention_mask = torch.tensor([1 for _ in range(len(ins))]).unsqueeze(0)
                inputs = {
                    "input_ids": input_ids.to(device),
                    "token_type_ids": token_type_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                }
                output = model(**inputs)
                probs = list(torch.nn.functional.softmax(output.logits[0], dim=-1))
            significance.append([probs[label_ind], ins, tokenized[i]])
        significance.sort()
        all_sig.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(significance[0][1][1:-1])))

        important_tokens.append(significance[0][2])
    important_df = pd.DataFrame({text_column: all_sig, label_column: train_df[label_column]})
    important_token_list = pd.DataFrame({"tokens": important_tokens})
    important_df.to_csv(out_dir1, index=False)
    important_token_list.to_csv(out_dir2, index=False)
    return important_df




def inspect_pos(dataframe, text_col = "tokens"):
    pos_dict = {"N": 0, "V":0, "J":0,}
    for i in tqdm(range(len(dataframe))):
        text = dataframe.iloc[i][text_col]
        tokenized = word_tokenize(text)
        pos_tagged = pos_tag(tokenized)
        poss = [pos for word,pos in pos_tagged]
        pos_count = Counter(poss)
        for k,v in pos_count.items():
            if k[0] in pos_dict:
                pos_dict[k[0]] += v
    return pos_dict


def collate_pos(dataset_list):
    agg_pos_list = []
    for dataset in dataset_list:
        dataset_path = f"./imp_stackoverflow_list.csv"
        dataframe = pd.read_csv(dataset_path)
        pos_dict = inspect_pos(dataframe)
        pos_list = [k for k,v in pos_dict.items()]
        for pos in pos_list:
            if pos not in agg_pos_list:
                agg_pos_list.append(pos)
    return agg_pos_list


def collate_counts(dataset_list):
    agg_pos_list = collate_pos(dataset_list)
    output_dict = {"pos" : agg_pos_list}
    agg_count_list = {k : 0 for k in agg_pos_list}
    for dataset in dataset_list:
        dataset_path = f"./imp_stackoverflow_list.csv"
        dataframe = pd.read_csv(dataset_path)
        pos_dict = inspect_pos(dataframe)
        sum_counts = sum([v for k,v in pos_dict.items()])
        count_list = []
        for pos in agg_pos_list:
            if pos in pos_dict:
                count_list.append(pos_dict[pos]/sum_counts)
                agg_count_list[pos] += pos_dict[pos]/sum_counts
            else:
                count_list.append(0)
                agg_count_list[pos] += 0
        output_dict[dataset] = count_list
    agg_counts = [v/len(dataset_list) for k,v in agg_count_list.items()]
    output_dict['counts'] = agg_counts

    output_df = pd.DataFrame(output_dict)
    output_df = output_df.sort_values(by=['counts'], ascending = False)
    output_df.to_csv("./table1.csv", index = False) # Code Run Test
                





def main(args):
    seed_everything()
    for dataset in args.datasets:
        dataframe = pd.read_csv(f"../dataset/{dataset}/train.csv")
        run_baseline(args, dataframe, dataset, feature=None, condition = None)
        out_dir1, out_dir2 = f'../dataset/{dataset}/imp_removed.csv', f'../dataset/{dataset}/imp_list.csv'
        make_important_tokens(dataframe, dataset, out_dir1, out_dir2)
    collate_counts(args.datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default = ["agnews","dbpedia","stackoverflow","banking","r8","ohsumed","amazon","yelp","imdb"])
    parser.add_argument("--model_name", default = "bert-base-uncased")
    parser.add_argument("--num_epochs", default = 20)
    parser.add_argument("--lr", default = 4e-5)
    parser.add_argument('--batch_size',  default = 128)
    parser.add_argument('--max_length',  default = 100)
    parser.add_argument('--pos', nargs= "+", default = ["noun", "verb", "adj"])
    args = parser.parse_args()
    main(args)