import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import torch

save_path = "./fast_data/"
topK = 16

os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path,"train"), exist_ok=True)
os.makedirs(os.path.join(save_path,"test"), exist_ok=True)
os.makedirs(os.path.join(save_path,"train_t"), exist_ok=True)
os.makedirs(os.path.join(save_path,"test_t"), exist_ok=True)
os.makedirs(os.path.join(save_path,"train_a"), exist_ok=True)
os.makedirs(os.path.join(save_path,"test_a"), exist_ok=True)

with open("./miniLM_L6_embs.pkl", "rb") as f:
    data = pickle.load(f)

train_posts = data["train_posts"]
train_mappings = data["train_mappings"]
train_tags = data["train_labels"]
train_embs = data["train_embs"]
train_timepoints = data["train_timepoints"]
train_analyzes = data["train_analyzes"]
train_ranks = data["train_ranks"]


test_posts = data["test_posts"]
test_mappings = data["test_mappings"]
test_tags = data["test_labels"]
test_embs = data["test_embs"]
test_timepoints = data["test_timepoints"]
test_analyzes = data["test_analyzes"]
test_ranks = data["test_ranks"]

for i, (mapping, label) in enumerate(zip(train_mappings, train_tags)):
    posts = train_posts[mapping]
    ranks = train_ranks[mapping]
    analyzes = train_analyzes[mapping]
    timepoints = [train_timepoints[i] for i in mapping]
    timepoints_ = np.array(timepoints)
    embs = train_embs[mapping,:]
    
    combined = list(zip(posts, ranks, analyzes))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    topk_idx = sorted_combined[:topK]
    topk_p = [item[0] for item in topk_idx]
    topk_r = [item[1] for item in topk_idx]
    topk_a = [item[2] for item in topk_idx]
    
    #raise Exception(embs.shape,timepoints_.shape)
    np.save(os.path.join(save_path,"train_t/{i:06}_{label}_emb.npy"),embs)
    np.save(os.path.join(save_path,"train_t/{i:06}_{label}.npy"),timepoints_)
    
    with open(os.path.join(save_path,"/train_a/{i:06}_{label}.txt"), "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in topk_a))
    with open(os.path.join(save_path,"/train/{i:06}_{label}.txt"), "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in topk_p))
        
for i, (mapping, label) in enumerate(zip(test_mappings, test_tags)):
    posts = test_posts[mapping]
    ranks = test_ranks[mapping]
    analyzes = test_analyzes[mapping]
    timepoints = [test_timepoints[i] for i in mapping]
    timepoints_ = np.array(timepoints)
    embs = test_embs[mapping,:]
    
    combined = list(zip(posts, ranks, analyzes))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    topk_idx = sorted_combined[:topK]
    topk_p = [item[0] for item in topk_idx]
    topk_r = [item[1] for item in topk_idx]
    topk_a = [item[2] for item in topk_idx]
    
    np.save(os.path.join(save_path,"test_t/{i:06}_{label}.npy"),timepoints_)
    np.save(os.path.join(save_path,"test_t/{i:06}_{label}_emb.npy"),embs)
    with open(os.path.join(save_path,"test_a/{i:06}_{label}.txt"), "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in topk_a))
    with open(os.path.join(save_path,"test/{i:06}_{label}.txt"), "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in topk_p))