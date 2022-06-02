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
from utils.utils import get_baseline_optimizer, get_pmixup_optimizer
import torch.nn.functional as F
from models.tmix import *


def run_baseline(dataframe, dataset_name, feature, lr, condition = None,text_column='text', label_column='label', model_name='bert-base-uncased', num_epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else cpu )
    lr = 4e-5
    max_length = 100
    batch_size = 124

    label_dict = get_label_dict(dataframe, label_column)
    train_dataset = TextDataset(dataframe, label_dict, text_column, label_column, max_length)
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

    ## baseline
    ## POS removed
    ## POS auged
    if not os.path.exists(f'../model_weights/{dataset_name}'):
        os.mkdir(f"../model_weights/{dataset_name}")
    if not feature:
        torch.save(model, f'../model_weights/{dataset_name}/baseline_model.pt')
    else:
        if condition:
            torch.save(model, f'../model_weights/{dataset_name}/model_{feature}_{condition}.pt')




def run_pmixup(train_df, dataset, feature,text_column="text", label_column="label", model_name = "bert-base-uncased",num_epochs = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else cpu )
    label_dict = get_label_dict(train_df, label_column)
    
    max_length_dict = {"dbpedia" : 128,
                       "oh" : 500,
                       "agnews": 400,
                       "stackoverflow" : 100,
                       "yelp": 500,
                        "banking" : 100,
                       "r8" : 500,
                       "imdb" : 500,
                      }
    
    batch_size_dict = {"dbpedia" : 196,
                       "oh" : 32,
                       "agnews" : 32,
                       "stackoverflow" : 196,
                       "yelp" : 32,
                       "banking" : 196,
                       "r8" : 64,
                       "imdb" : 32
                      }
    

    max_length = 100
    batch_size = 32
    
    print(f"max_length : {max_length}, batch_Size : {batch_size}")
    train_dataset = TextDataset(train_df, label_dict, text_column, label_column, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = MixText(num_labels = len(label_dict), mix_option = True).cuda()
    model = nn.DataParallel(model)
    lr = 3e-5
    optimizer = get_pmixup_optimizer(model, lr)


    mix_layer_set = [7,9,12]

    for epoch in range(num_epochs):
        alpha = 16
        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)
        #print(l)
        mix_layer = np.random.choice(mix_layer_set,1)[0]
        mix_layer = mix_layer - 1
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            print(batch['input_ids'].size())
            idx = torch.randperm(batch['input_ids'].size(0))
            inputs1 = batch['input_ids']
            inputs2 = torch.index_select(inputs1.cpu(),dim=0, index= idx)
            inputs = {}
            inputs['x'] = inputs1.squeeze(1).to(device)
            inputs['x2'] = inputs2.squeeze(1).to(device)
            inputs['l'] = l 
            outputs = model(**inputs, mix_layer=mix_layer)

            real_labs = batch['labels'].cuda()

            targets_x = torch.zeros(batch['input_ids'].size(0), torch.tensor(len(label_dict))).cuda().scatter_(1, real_labs.cuda().view(-1,1),1)

            out_labs = torch.index_select(targets_x,dim=0, index=idx.cuda())

            mixed_target = l * targets_x + (1-l)*out_labs

            Lx = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1)*mixed_target.cuda(), dim=1))
            loss = Lx

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model, f'../model_weights/{dataset}/pmixup_model_{feature}.pt')