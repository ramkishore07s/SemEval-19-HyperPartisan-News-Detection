
# coding: utf-8

# In[5]:

import torch
import _pickle as pickle
import random


# In[6]:

get_ipython().magic(u'run Model.ipynb')


# In[9]:

get_ipython().magic(u'run Helpers.ipynb')


# In[10]:

model = Model(NO_URLS, EMB)
model.cuda()


# In[11]:

model.load_state_dict(torch.load('parameters_1'))


# In[12]:

torch.cuda.set_device(0)


# In[13]:

validation_data = pickle.load(open('validation_0', 'rb+'))
#validation_data.extend(pickle.load(open('validation_1', 'rb+')))


# In[14]:

batch_size = 20


# In[15]:

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


# In[19]:

bias_tp, bias_tn, bias_fp, bias_fn = torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0])
truth_tp, truth_tn, truth_fp, truth_fn = torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0])


# In[74]:

bias = []
p_bias = []


# In[75]:

with torch.no_grad():
    for batch in batches:
        if len(batch) == 20:
            input_ = torch.cuda.LongTensor([lines[i] for i in batch])
            urls_ = torch.cuda.LongTensor([urls[i] for i in batch])
            titles_ = torch.cuda.LongTensor([titles[i] for i in batch])
            truth_ = torch.cuda.FloatTensor([[truth[i]] for i in batch])
            bias_ = torch.max(torch.cuda.FloatTensor([class_[i] for i in batch]), dim=1)[1]
            pbias, ptruth = model(input_, urls_, titles_)
            
            truth_tp += torch.sum(ptruth.gt(.5) * truth_.gt(0)).float()
            truth_tn += torch.sum(ptruth.le(.5) * truth_.le(0)).float()
            truth_fp += torch.sum(ptruth.gt(.5) * truth_.le(0)).float()
            truth_fn += torch.sum(ptruth.le(.5) * truth_.gt(0)).float()
            
            pbias = torch.max(pbias, dim=1)[1]
            bias.extend(np.array(bias_.cpu().data))
            p_bias.extend(np.array(pbias.cpu().data))
        print(batches.index(batch), end='\r')


# In[76]:

truth_tp, truth_tn, truth_fp, truth_fn


# In[77]:

precision = truth_tp / (truth_tp + truth_fp)
recall = truth_tp / (truth_tp + truth_fn)
f1_score = 2 * precision * recall / (precision + recall)
print(f1_score)


# In[78]:

bias = np.array(bias)
p_bias = np.array(p_bias)


# In[83]:

confusion_matrix = np.zeros((5, 5))


# In[84]:

for i, j in zip(bias, p_bias):
    confusion_matrix[i][j] += 1


# In[85]:

confusion_matrix


# In[ ]:



