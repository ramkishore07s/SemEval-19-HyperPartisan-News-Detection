import os
import re
import csv
import numpy as np 
import nltk
import pickle
import imp
from nltk.corpus import stopwords


reg = ["https?\040:\040\/(?:\040{0,2}[\/=\?%]\040{0,2}[A-Za-z0-9\.\-_]+)*",
       "\/r\/[^\040\n]*",
       "[:;<]\040{0,1}[/)/(|/\\3]",
       "[a-z]+\040n't","[a-zA-Z]+\040{0,1}'[a-zA-Z]+",
       "[a-zA-Z]+\-[a-zA-Z]+",
       "[a-zA-Z0-9]+"]

# vocab = []


def tokenise(iterator, string):
    if iterator == len(reg) - 1:
        return re.findall(reg[iterator], string)
    tok_split = re.compile(reg[iterator]).split(string)
    temp_tok = re.findall(reg[iterator], string)
    tokens = []
    for tnk in tok_split:
        new_it = iterator + 1
        each = tokenise(new_it, tnk)
        tokens += each
        if len(temp_tok) > 0:
            tokens.append(temp_tok.pop(0))
    return tokens

def process_train(folder, cnt=10000):
    count = 1
    titles = []
    url = []
    body = []
    for file in os.listdir(folder):
        # print(count,file)
        process_file(folder + file,titles,url,body)
        if count >cnt:
            break
        count += 1
    return titles,url,body

def process_labels(file,count=10000):
    #extract the ground-truth and bias from the labels
    truth = []
    bias = []
    with open(file,'r',encoding='UTF-8') as f:
        f = csv.reader(f,delimiter='\n')
        cnt = 0
        for each in f:
            if each != '':
                split_label = each[0].split(',')
                truth.append(split_label[1])
                bias.append(split_label[2])
                cnt+=1
            if cnt > count: #temp statement
                break

    return truth,bias 

def remove_stop(text):
    stop_words = set(stopwords.words('english'))
    text_list = [word for word in text if not word in stop_words]
    # for i in text_list:
    #     vocab.append(i)
    return(' '.join(text_list))

def process_file(file,title,url,body):
    
    f = open(file,encoding='utf-8').read()
    f_split = f.lower().split("\n\n")
    title_untok = f_split[0]
    body_untok = f_split[2]
    body_tok = tokenise(0,body_untok)
    title_tok = tokenise(0,title_untok)
    title.append(remove_stop(title_tok))
    body.append(remove_stop(body_tok))
    

