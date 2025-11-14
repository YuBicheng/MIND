# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import xml.dom.minidom
import string
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch


# %%
sbert = SentenceTransformer('/mnt/proj/paraphrase-MiniLM-L6-v2')


# %%
def get_input_data_train(file, path):
    post_num = 0
    dom = xml.dom.minidom.parse(path + "/" + file)
    collection = dom.documentElement
    title = collection.getElementsByTagName('TITLE')
    text = collection.getElementsByTagName('TEXT')
    date = collection.getElementsByTagName('DATE')
    posts = []
    timepoints = []
    analyzes = []
    for i in range(len(title)):
        post = title[i].firstChild.data + ' ' + text[i].firstChild.data
        post = re.sub('\n', ' ', post)
        if len(post) > 0:
            posts.append(post.strip())
            post_num = post_num + 1
            
            time = re.search(r'(\d{2}):(\d{2}):\d{2}',date[i].firstChild.data)
            hour = int(time.group(1))
            minute = int(time.group(2))
            timepoints.append([hour,minute])
    return posts,timepoints,  post_num

def get_input_data_test(file, path):
    post_num = 0
    dom = xml.dom.minidom.parse(path + "/" + file)
    collection = dom.documentElement
    title = collection.getElementsByTagName('TITLE')
    text = collection.getElementsByTagName('TEXT')
    date = collection.getElementsByTagName('DATE')
    analyze = collection.getElementsByTagName('ANALYZE')
    rank = collection.getElementsByTagName('RANK')
    if len(analyze) != len(title):
        raise Exception(file)
    posts = []
    timepoints = []
    analyzes = []
    ranks = []
    for i in range(len(title)):
        post = title[i].firstChild.data + ' ' + text[i].firstChild.data
        post = re.sub('\n', ' ', post)
        if len(post) > 0:
            posts.append(post.strip())
            post_num = post_num + 1
            
            time = re.search(r'(\d{2}):(\d{2}):\d{2}',date[i].firstChild.data)
            hour = int(time.group(1))
            minute = int(time.group(2))
            timepoints.append([hour,minute])
            #print(analyze[i],file)
            analyzes.append(analyze[i].firstChild.data)
            ranks.append(rank[i].firstChild.data)
    return posts,timepoints, analyzes,ranks, post_num

# %%
train_posts = []
train_tags = []
train_mappings = []
train_timepoints = []
train_analyzes = []
train_ranks = []

test_posts = []
test_timepoints = []
test_analyzes = []
test_ranks = []
test_tags = []
test_mappings = []
for base_path in ["negative_examples_anonymous_a", "negative_examples_test_a", "positive_examples_anonymous_a", "positive_examples_test_a"]:
    base_path = "/mnt/proj/llamps-dataset/"+base_path
    filenames = sorted(os.listdir(base_path))
    for fname in filenames:
        #posts,timepoints, analyzes, post_num = get_input_data(fname, base_path)
        if "anonymous" in base_path:
            posts,timepoints, analyzes, ranks, post_num = get_input_data_test(fname, base_path)
            train_mappings.append(list(range(len(train_posts), len(train_posts)+post_num)))
            train_posts.extend(posts)
            train_timepoints.extend(timepoints)
            train_analyzes.extend(analyzes)
            train_ranks.extend(ranks)
            train_tags.append(int("positive" in base_path))
        else:
            posts,timepoints, analyzes, ranks, post_num = get_input_data_test(fname, base_path)
            test_mappings.append(list(range(len(test_posts), len(test_posts)+post_num)))
            test_posts.extend(posts)
            test_timepoints.extend(timepoints)
            test_analyzes.extend(analyzes)
            test_ranks.extend(ranks)
            test_tags.append(int("positive" in base_path))


# %%
train_embs = sbert.encode(train_posts, convert_to_tensor=False)
train_embs.shape


test_embs = sbert.encode(test_posts, convert_to_tensor=False)
test_embs.shape


with open("./miniLM_L6_embs.pkl", "wb") as f:
    pickle.dump({
        "train_posts": train_posts,
        "train_mappings": train_mappings,
        "train_labels": train_tags,
        "train_embs": train_embs,
        "train_timepoints":train_timepoints,
        "train_analyzes":train_analyzes,
        "train_ranks":train_ranks,
        "test_posts": test_posts,
        "test_mappings": test_mappings,
        "test_labels": test_tags,
        "test_embs": test_embs,
        "test_timepoints":test_timepoints,
        "test_analyzes":test_analyzes,
        "test_ranks":test_ranks
    }, f)

