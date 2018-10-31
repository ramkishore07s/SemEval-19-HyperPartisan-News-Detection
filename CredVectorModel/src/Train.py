
# coding: utf-8

# In[3]:

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


# In[4]:

import _pickle as pickle
import random


# In[5]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:

torch.cuda.set_device(0)


# In[21]:

#get_ipython().magic(u'run Model.ipynb')
from Model import *

# In[8]:

#get_ipython().magic(u'run Helpers.ipynb')
from Helpers import *

# In[15]:

use_source_bias = False
no_iterations = 10


# In[34]:

class TrainModel():
    def __init__(self, model):
        self.model = model
        self.raw_data, self.lines, self.pad_lengths, self.truth, self.class_, self.urls = None, None, None, None, None, None
        self.file_no = 0
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
                    bias = torch.cuda.ByteTensor([self.class_[i] for i in batch])
                    lengths = torch.cuda.FloatTensor([self.pad_lengths[i] for i in batch])
                    pbias, ptruth = self.model(input, urls, titles) #lengths)
                    # normal BCE loss
                    #  loss1 = self.loss2(ptruth, truth)
                    # softmax loss
                    pbias_select = pbias.masked_select(bias)
                    ones = torch.ones(pbias_select.size()).cuda()
                    loss2 = self.loss2(pbias_select, ones)
                    self.output = [pbias, pbias_select, ones]
                    # add both losses
                    loss = loss2 # + loss1
                    print(str(self.batches.index(batch)) + ' ' + str(loss), end='\r')
                    self.losses.append(loss.cpu().data)
                    loss.backward()
                    self.optimizer.step()


# In[35]:

m = Model(NO_URLS, EMB, use_source_bias=use_source_bias)
m.cuda()


# In[36]:

t = TrainModel(m)


# In[37]:

for i in range(no_iterations):
    t.train_epoch()
    torch.save(t.model.state_dict(), 'parameters_' + str(i))
    # add validation here

