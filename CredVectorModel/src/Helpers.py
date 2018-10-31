import _pickle as pickle
import torch
import numpy as np
# coding: utf-8

# In[ ]:

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


# In[ ]:

def pad_titles(data, max_sen_len=20, padding_idx=400003):
    lines = []
    for row in data:
        line = row[0]
        if len(line) > max_sen_len:
            lines.append(line[0:max_sen_len])
        else:
            lines.append(line + [padding_idx for _ in range(max_sen_len - len(line))])
    return lines


# In[ ]:

class_mapping = {
    'left': [1, 0, 0, 0, 0],
    'left-center': [0, 1, 0, 0, 0],
    'least': [0, 0, 1, 0, 0],
    'right-center': [0, 0, 0, 1, 0],
    'right': [0, 0, 0, 0, 1],
}


# In[ ]:

EMB = pickle.load(open('./Meta/glove_100d_vectors', 'rb'))
EMB.append(np.zeros(100))
EMB = torch.Tensor(EMB)


# In[ ]:

URLS = pickle.load(open('./Meta/urls', 'rb'))
NO_URLS = len(URLS)

