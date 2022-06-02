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
from utils.utils import seed_everything


def tmix(train_df, text_column,label_name,dataset,model_name="bert-base-uncased",num_epochs=30,save=True,syntax=False,):
    device = torch.device("cuda")
    label_dict = get_label_dict(train_df, label_column)

    max_length = 100
    batch_size = 124

    train_dataset = TextDataset(
            train_df, label_dict, text_column, label_name, max_length
        )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = MixText(num_labels=len(label_dict), mix_option=True).cuda()
    model = nn.DataParallel(model)

    lr = 3e-5
    optimizer = get_pmixup_optimizer(model, lr)
    mix_layer_set = [7, 9, 12]

    train_criterion = SemiLoss()

    for epoch in range(num_epochs):
        alpha = 16
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)
        mix_layer = np.random.choice(mix_layer_set, 1)[0]
        mix_layer = mix_layer - 1
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            idx = torch.randperm(batch["input_ids"].size(0))
            inputs1 = batch["input_ids"]
            inputs2 = torch.index_select(inputs1.cpu(), dim=0, index=idx)
            inputs = {}
            inputs["x"] = inputs1.squeeze(1).to(device)
            inputs["x2"] = inputs2.squeeze(1).to(device)
            inputs["l"] = l
            outputs = model(**inputs, mix_layer=mix_layer)

            real_labs = batch["labels"].cuda()

            targets_x = (
                torch.zeros(batch["input_ids"].size(0), torch.tensor(len(label_dict)))
                .cuda()
                .scatter_(1, real_labs.cuda().view(-1, 1), 1)
            )

            out_labs = torch.index_select(targets_x, dim=0, index=idx.cuda())

            mixed_target = l * targets_x + (1 - l) * out_labs
            Lx = -torch.mean(
                torch.sum(F.log_softmax(outputs, dim=1) * mixed_target.cuda(), dim=1)
            )
            loss = Lx
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



def main():
    ## IMP, POS가 없다면 IMP, POS AUG 만들기
    

seed_everything()

for dataset in NOT_TRAIN:
    if dataset in NOT_TRAIN:
        project_name = dataset.split("/")[2]
        for data_size in ["10"]:
            if os.path.exists(dataset + f"/train_{data_size}.csv"):

                train_df, test_df = open_file(
                    glob(dataset + f"/train_{data_size}.csv")[0]
                ), open_file(glob(dataset + "/test.*sv")[0])
                text_column, label_column = train_df.columns[0], train_df.columns[1]
                wandb.init(project = str(dataset.split("/")[2] + "_final"),
                    entity="so_lucci",
                    name=project_name + f"_{data_size}_baseline_bert",
                )


                for pos in ["adj", "noun", "verb"]:
                    train_df, test_df = open_file(
                        glob(dataset + f"/train_{data_size}_{pos}_aug.csv")[0]
                    ), open_file(glob(dataset + "/test.*sv")[0])
                    if glob(dataset + "/dev.*sv"):
                        dev_df = open_file(glob(dataset + "/dev.*sv")[0])
                        train_df.append(dev_df)
                    else:
                        text_column, label_column = train_df.columns[0], train_df.columns[1]
                    print(label_column)

                    wandb.init(
                        project=str(dataset.split("/")[2] + "_final"),
                        entity="so_lucci",
                        name=project_name + f"_{data_size}_{pos}_tmix",
                    )
                    # run_baseline(train_df, test_df, text_column, label_column, project_name)
                    tmix(
                        train_df,
                        test_df,
                        text_column,
                        label_column,
                        project_name,
                        num_epochs=30,
                    )