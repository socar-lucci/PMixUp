import argparse
import pandas as pd
import os
from collections import Counter
from tqdm import tqdm
import os
import random
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download("omw-1.4")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


def inspect_pos(dataframe, text_col = "text"):
    pos_dict = {"N": 0, "V":0, "J":0, "C": 0, "D":0, "PR":0, "M": 0, "WP": 0, "RB" : 0}
    for i in tqdm(range(len(dataframe))):
        text = dataframe.iloc[i][text_col]
        tokenized = word_tokenize(text)
        pos_tagged = pos_tag(tokenized)
        poss = [pos for word,pos in pos_tagged]
        pos_count = Counter(poss)
        for k,v in pos_count.items():
            if k[0] in pos_dict:
                pos_dict[k[0]] += v
            else:
                pos_dict[k] = v
    return pos_dict


def collate_pos(dataset_list):
    agg_pos_list = []
    for dataset in dataset_list:
        dataset_path = f"../dataset/{dataset}/train.csv"
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
        dataset_path = f"../dataset/{dataset}/train.csv"
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
    if not os.path.exists("../outputs"):
        os.mkdir("../outputs")
    output_df.to_csv("../output/table_1.csv", index = False)
                




def main(args):
    print("----- Inspecting file -----")
    collate_counts(datasets)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default = ["agnews","dbpedia","stackoverflow","banking","r8","ohsumed","amazon","yelp","imdb"])
    args = parser.parse_args()
    main(args)