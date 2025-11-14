import os
import re
import heapq
import copy
from tqdm import tqdm
import xml.etree.ElementTree as ET

from GPTAnalyze import  send_message_a,testfunc
from GPTRank import  send_message_r
#input: str output "score:3.5"

"""
将用户的每条发布内容的潜在心理问题写入并重新保存在_a文件夹中
"""


def exist(file_name,save_path:str):
    filenames = sorted(os.listdir(save_path))
    
    if file_name in filenames:
        return True
    else:
        return False

def userpost_select(file_name:str,path:str,):
    save_path = path+"_a"
    if exist(file_name,save_path):
        return None
    tree = ET.parse(os.path.join(path,file_name))
    root = tree.getroot()
    posts = root.findall('WRITING')


    for post in posts:
        text = post.find('TEXT')
        date = post.find('DATE')
        text = text.text
        rank = send_message_r(text)
        if len(text)>10:
            if len(text)>512:
                text = text[:512]
            analyze = send_message_a(text)
        else:
            analyze = "None"
        element_r = ET.Element('RANK')
        element_r.text = rank
        element_a = ET.Element('ANALYZE')
        element_a.text = analyze
        post.append(element_a)
    tree.write(os.path.join(save_path,file_name))
        

def main():
    path = r"./negative_examples_test"
    save_path = path+'_a'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filenames = sorted(os.listdir(path))
    for file in filenames:
        userpost_select(file,path)
        print(file)
    #userpost_select(os.path.join(path,"train_subject1555_10.xml"),'1555','1')

if __name__=="__main__":
    main()
    #userpost_select("test_subject25.xml",r"./negative_examples_test")