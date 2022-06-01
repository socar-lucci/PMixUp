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
        syns, cnt = synonym_count(word)
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
        text, label = data['text'].lower(), data['label']
        splitted_text = text.split()
        imp_tok = imp_tokens[ind]
        if imp_tok in yess:
            synss = list(set(syn_dict[imp_tok]))
            for syn in synss[:2]:
                tmp_text = text
                nt = tmp_text.replace(imp_tok, syn)
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
    auged_df.to_csv("../dataset/stackoverflow/auged_train.csv", index= False)
            

def make_mini_sample(train_df, dataset,sample_size):
    sample_texts, sample_labels = [], []
    for key, values in tqdm(Counter(train_df['label']).items()):
        tmp = [train_df.iloc[i]['text'] for i in range(len(train_df)) if train_df.iloc[i]['label'] == key]
        random_texts = random.sample(tmp, sample_size)
        sample_texts += random_texts
        labs = [key for _ in range(sample_size)]
        sample_labels += labs
    new_df = pd.DataFrame({"text": sample_texts, "label": sample_labels})
    new_df.to_csv(f"../dataset/{dataset}/train_{sample_size}.csv")



def pos_augmentation(train_df):
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
            
        nouns = [[i,word[0]] for i,word in enumerate(tagged) if word[1].startswith("V")]
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
    pos_removed = pd.DataFrame({"text": ni_aug, "label" : ni_lab})
            
    pos_removed = pd.DataFrame({"text": ni_aug, "label" : ni_lab})
    pos_removed.to_csv("../dataset/stackoverflow/train_verb_aug.csv", index=False)


def run_baseline(train_df, dataset_name, text_column='text', label_column='label', model_name='bert-base-uncased', num_epochs=1):
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
    
    if not os.path.exists('../model_weights'):
        os.mkdir("../model_weights/")
    torch.save(model, f'../model_weights/model_{dataset_name}_.pt')







def main():
    train_df = pd.read_csv("../dataset/stackoverflow/train.csv")[:100]
    imp_list = pd.read_csv("../dataset/stackoverflow/imp_list.csv")['tokens'].tolist()
    important_augmentation(train_df, imp_list)
    pos_augmentation(train_df)
    #run_baseline()

    #make_mini_sample(train_df, "stackoverflow",10)
    #for files in [10]:
    #    df = pd.read_csv(f"../dataset/stackoverflow/train_{files}.csv")
    #    important_augmentation(train_df)
    #    pos_augmentation()
    #    run_baseline(important_aug_file)
    #    for pos in ["n", "v", "adj"]:
    #        run_baseline(pos_aug)



if __name__ == "__main__":
    main()