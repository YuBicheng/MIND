from collections import defaultdict
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy.io import loadmat
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from collections import defaultdict

class FlatDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train"):
        assert split in {"train", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split)
        for fname in os.listdir(input_dir2):
            #print(fname)
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            sample = {}
            sample["text"] = open(os.path.join(input_dir2, fname), encoding="utf-8").read()
            tokenized = tokenizer(sample["text"], truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            self.labels.append(label)

    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_flat(data):
    labels = []
    processed_batch = defaultdict(list)
    for item, label in data:
        for k, v in item.items():
            processed_batch[k].append(v)
        labels.append(label)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        processed_batch[k] = torch.LongTensor(processed_batch[k])
    labels = torch.FloatTensor(labels)
    return processed_batch, labels

class FlatDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "train")
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")
        elif stage == "test":
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_flat, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_flat, pin_memory=True, num_workers=4)

class HierDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", max_posts=64):
        #input_dir = ./processed/combined_maxsim16
        assert split in {"train", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        # hparam of timepoints 
        self.time_dim = 768
        assert self.time_dim%24==0,'time dim should be '
        self.hour_span = self.time_dim//24
        self.minutes_span = 60//self.hour_span
        self.half_time_dim = self.time_dim//2
        # Normal distribution
        sigma = 1
        #sigma = 0.03 # old version
        x = np.arange(-self.time_dim/2, self.time_dim/2, 1)
        y = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma, -1), np.exp(-np.power(x, 2) / 2 * sigma ** 2))
        self.norm = torch.from_numpy(y).to(torch.float32)
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split)
        anaylize_dir = os.path.join(input_dir, split+'_a')
        timepoints_dir = os.path.join(input_dir,split+'_t')
        for fname in tqdm(os.listdir(input_dir2)):
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            sample = {}
            posts = open(os.path.join(input_dir2, fname), encoding="utf-8").read().strip().split("\n")[:max_posts]
            analyze = open(os.path.join(anaylize_dir, fname), encoding="utf-8").read().strip().split("\n")[:max_posts]
            tokenized = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len)
            tokenized_a = tokenizer(analyze, truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            for k, v in tokenized_a.items():
                sample[k+'_a'] = v
            tfname = fname[:-3]+"npy"
            embfname = fname[:-4]+"_emb.npy"
            embs = np.load(os.path.join(timepoints_dir,embfname))
            timepoints = np.load(os.path.join(timepoints_dir,tfname))
            timepoints_emb = self.norm_emb(torch.from_numpy(timepoints))
            sample['timepoints_emb'] = timepoints_emb
            sample['embs'] = torch.from_numpy(embs)
            sample['timeindex'] = self.timeindexs(torch.from_numpy(timepoints))
            #raise Exception(sample['timeindex'].dtype,sample['timeindex'].shape)
            self.data.append(sample)
            self.labels.append(label)
    def timeindexs(self,timepoints):
        hour_index = timepoints[:,0]
        time_index = hour_index
        return time_index.long()
        
        
    # embedding person's timepoins [[1,34],...[23,59]]-> vector(768)
    def norm_emb(self,timepoints:torch.tensor)->torch.tensor:
        #norm_x = torch.arange(-self.time_dim/2,self.time_dim/2,1)
        #raise Exception(norm_x) #[-xxx,...,xxx-1]
        
        #index = hour * hour_span + minute//minutes_span
        # raise Exception("shape,dtype", timepoints.shape,timepoints.dtype) #('shape', (1131, 2)) int64
        indexs = timepoints[:,0] * self.hour_span + torch.floor(timepoints[:,1] / self.minutes_span).to(torch.int)
        indexs = indexs.to(torch.int)
        #raise Exception(indexs.max(),indexs.min())
        embs = torch.zeros(0,self.time_dim)
        #embs = torch.zeros(self.time_dim)
        for index in indexs:
            timeline = self.norm
            if index<self.half_time_dim:
                cut = self.half_time_dim-index
                left = self.norm[:cut]
                right = self.norm[cut:]
                timeline = torch.cat((right,left),dim=0)
            elif index> self.half_time_dim:
                cut = index-self.half_time_dim
                left = self.norm[:self.time_dim-cut]
                right = self.norm[self.time_dim-cut:]
                timeline = torch.cat((right,left),dim=0)
            #embs = embs + timeline
            embs = torch.cat((embs,timeline.unsqueeze(0)),dim=0)
        #return embs/(timepoints.size(0)/2)
        return embs
        
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_hier(data):
    labels = []
    processed_batch = []
    for item, label in data:
        user_feats = {}
        #raise Exception(item.items())
        for k, v in item.items():
            if k in ["timepoints_emb","embs"]:
                user_feats[k] = v
                continue
            user_feats[k] = torch.LongTensor(v)
        processed_batch.append(user_feats)
        labels.append(label)
    labels = torch.FloatTensor(np.array(labels))
    return processed_batch, labels

class HierDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "train")
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test")
        elif stage == "test":
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_hier, shuffle=True, pin_memory=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=0)



def infer_preprocess(tokenizer, texts, max_len,timepoints_emb,embs,timeinex,analyze):
    batch_ = {}
    batch = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        batch_[k] = torch.LongTensor(batch[k])
    batch_analyze = tokenizer(analyze, truncation=True, padding='max_length', max_length=max_len)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        batch_[k+'_a'] = torch.LongTensor(batch_analyze[k])
    batch_['timepoints_emb'] = timepoints_emb
    batch_['embs'] = embs
    batch_['timeindex'] = timeinex
    return batch_
