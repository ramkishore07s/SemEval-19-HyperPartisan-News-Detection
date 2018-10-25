
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


# In[2]:

import _pickle as pickle
import random


# In[3]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:

torch.cuda.set_device(1)


# In[5]:

get_ipython().magic(u'run Model.ipynb')


# In[6]:

emb = pickle.load(open('./Meta/glove_100d_vectors', 'rb'))
emb.append(np.zeros(100))
emb = torch.Tensor(emb)


# In[7]:

urls = pickle.load(open('./Meta/urls', 'rb'))
no_urls = len(urls)


# In[8]:

class_mapping = {
    'left': [1, 0, 0, 0, 0],
    'left-center': [0, 1, 0, 0, 0],
    'least': [0, 0, 1, 0, 0],
    'right': [0, 0, 0, 0, 1],
    'right-center': [0, 0, 0, 1, 0],
}


# In[9]:

def pad_data(data, max_sen_len=50, max_doc_len=90, padding_idx=400003):
    lines = []
    lengths = []
    for doc in data:
        lines.append([])
        lengths.append([])
        for line in doc[2][0:max_doc_len]:
            if len(line) > max_sen_len:
                line = line[0:max_sen_len]
                lengths[-1].append(max_sen_len)
            else:
                lengths[-1].append(len(line))
                line = line + [padding_idx for _ in range(max_sen_len - len(line))]
            lines[-1].append(line)
    return lines, lengths


# In[10]:

def pad_titles(data, max_sen_len=20, padding_idx=400003):
    lines = []
    for row in data:
        line = row[0]
        if len(line) > max_sen_len:
            lines.append(line[0:max_sen_len])
        else:
            lines.append(line + [padding_idx for _ in range(max_sen_len - len(line))])
    return lines


# In[11]:

class TrainModel():
    def __init__(self, model):
        self.model = model
        self.raw_data, self.lines, self.pad_lengths, self.truth, self.class_, self.urls = None, None, None, None, None, None
        self.file_no = 1
        self.max_file_no = 8
        self.batch_size = 20
        
        self.loss1 = nn.BCEWithLogitsLoss()
        self.loss2 = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.losses = []
        
    def __get_next_batch(self):
        pass
    
    def __load_next(self):
        self.raw_data = pickle.load(open('training_' + str(self.file_no), 'rb'))
        self.lines, self.pad_lengths = pad_data(self.raw_data)
        self.class_ = [class_mapping[row[-1]] for row in self.raw_data]
        self.truth = [int(row[-2]) for row in self.raw_data]
        self.lengths = [[] for _ in range(90)]
        for i, j in enumerate(self.lines):
            if not len(j) == 0:
                self.lengths[len(j) - 1].append(i)
        self.urls = [i[1][0:90] for i in self.raw_data]
        self.batches = []
        for unit in self.lengths:
            for i in range(0, len(unit), self.batch_size):
                self.batches.append(unit[i:i+self.batch_size])
        self.titles = pad_titles(self.raw_data)
        random.shuffle(self.batches)
        self.file_no += 1
                
    def train_epoch(self):
        while self.file_no < self.max_file_no:
            self.__load_next()
            for batch in self.batches:
                if len(batch) == 20:
                    self.model.zero_grad()
                    input = torch.cuda.LongTensor([self.lines[i] for i in batch])
                    urls = torch.cuda.LongTensor([self.urls[i] for i in batch])
                    titles = torch.cuda.LongTensor([self.titles[i] for i in batch])
                    truth = torch.cuda.FloatTensor([[self.truth[i]] for i in batch])
                    bias = torch.cuda.FloatTensor([self.class_[i] for i in batch])
                    pbias, ptruth = self.model(input, urls, titles)
                    self.output = pbias, bias
                    loss1 = self.loss2(ptruth, truth)
                    loss2 = self.loss2(pbias, bias)
                    loss = loss1 + loss2
                    print(str(self.batches.index(batch)) + ' ' + str(loss), end='\r')
                    self.losses.append(loss.cpu().data)
                    loss.backward()
                    self.optimizer.step()


# In[12]:

m = Model(no_urls, emb)
m.cuda()


# In[13]:

t = TrainModel(m)


# In[ ]:

t.train_epoch()


# In[ ]:

import matplotlib.pyplot as plt


# In[ ]:

plt.plot(t.losses)


# **RESULTS ON VALIDATION SET**

# In[ ]:

validation_data = pickle.load(open('validation_0', 'rb+'))
validation_data.extend(pickle.load(open('validation_1', 'rb+')))


# In[ ]:

batch_size = 20


# In[ ]:

raw_data = validation_data
lines, pad_lengths = pad_data(raw_data)
class_ = [class_mapping[row[-1]] for row in raw_data]
truth = [int(row[-2]) for row in raw_data]
lengths = [[] for _ in range(90)]
for i, j in enumerate(lines):
    if not len(j) == 0:
        lengths[len(j) - 1].append(i)
urls = [i[1][0:90] for i in raw_data]
batches = []
for unit in lengths:
    for i in range(0, len(unit), batch_size):
        batches.append(unit[i:i+batch_size])
titles = pad_titles(raw_data)
random.shuffle(batches)


# In[ ]:

bias_tp, bias_tn, bias_fp, bias_fn = 0, 0, 0, 0
truth_tp, truth_tn, truth_fp, truth_fn = 0, 0, 0, 0


# In[ ]:

for batch in batches:
    if len(batch) == 20:
        t.model.zero_grad()
        input_ = torch.cuda.LongTensor([lines[i] for i in batch])
        urls_ = torch.cuda.LongTensor([urls[i] for i in batch])
        titles_ = torch.cuda.LongTensor([titles[i] for i in batch])
        truth_ = torch.cuda.FloatTensor([[truth[i]] for i in batch])
        bias_ = torch.cuda.FloatTensor([class_[i] for i in batch])
        pbias, ptruth = t.model(input_, urls_, titles_)
        
        truth_tp += torch.sum(pbias.gt(.5) * bias.gt(0)).cpu().data
        truth_tn += torch.sum(pbias.le(.5) * bias.le(0)).cpu().data
        truth_fp += torch.sum(pbias.gt(.5) * bias.le(0)).cpu().data
        truth_fn += torch.sum(pbias.le(.5) * bias.gt(0)).cpu().data


# In[ ]:

precision = truth_tp.long() / (truth_tp + truth_fp).long()
recall = truth_tp.long() / (truth_tp + truth_fn).long()


# In[ ]:

f1_score = 2 * precision * recall / (precision + recall)


# In[ ]:

f1_score

