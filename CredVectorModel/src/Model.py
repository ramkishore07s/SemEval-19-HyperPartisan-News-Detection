
# coding: utf-8

# # HyperPartisan News Detection
# 
# **News articles in this Dataset cite various sources throughout. We can leverage this information when constructing representation for sentences.**

# Let representation of a given sentence `s` be `h(s)` and assume that `s` had a hyperlink to some link `k`, meaning the information `s` was obtained from source `k`. 
# 
# Then we can modify the representation as:
# $$SentenceRepresentation(s) = tanh(W_{k}(h(s)) + bias_{k})$$
# 
# If a sentence does not have any citations, then a default Matrix is used.
# 
# An alterate approach could be to have a weightage vector for each source.

# In[2]:

import numpy as np


# In[3]:

import torch
import torch.nn as nn
import torch.nn.functional as f


# In[4]:

from config import model_config as config


# In[5]:

def random_(size):
    return np.random.uniform(1, -1, size)


# In[6]:

def create_embedding_layer(weights, non_trainable=False, padding_idx=401005):
    if weights is not None:
        emb_len, word_dims = weights.size()
        emb_layer = torch.nn.Embedding(emb_len, word_dims, padding_idx=emb_len-1)
        emb_layer.load_state_dict({'weight': weights})
    else:
        emb_layer = torch.nn.Embedding(10, config.word_emb_size, padding_idx=9)
        emb_len, word_dims = 10, config.word_emb_size
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, emb_len, word_dims


# In[7]:

class LSTMSentenceEncoderParallel(nn.Module):
    '''
    INPUT: 3D Tensor of word Ids (batch_size * no_sentences_per_doc * no_words_per_sen)
    OUTPUT: 3D Tensor of sentence Embeddings (batch_size * no_sentence_per_doc * sen_emb_size)
    '''
    def __init__(self, weights=None):

        super(LSTMSentenceEncoderParallel, self).__init__()
        self.embeddings, vocab, emb_len = create_embedding_layer(weights, config.word_emb_size)
        self.sentenceEncoder = nn.LSTM(config.word_emb_size, 
                                       config.sen_emb_size, 
                                       batch_first=True, 
                                       bidirectional=config.sen_bidirectional)
        self.sen_emb_size = config.sen_emb_size
        if config.sen_bidirectional: self.sen_emb_size *= 2

    def forward(self, input, sen_len=config.sen_len):
        words = self.embeddings(input.view(-1)).view(-1, sen_len, config.word_emb_size)
        hn = self.sentenceEncoder(words)[1][0]
        sentences = torch.cat((hn[0], hn[1]), dim=1).reshape(config.batch_size, -1, self.sen_emb_size)
        return sentences


# In[8]:

class LSTMSentenceEncoderPadding(nn.Module):
    '''
    INPUT: 3D Tensor of word Ids (batch_size * no_sentences_per_doc * no_words_per_sen)
    OUTPUT: 3D Tensor of sentence Embeddings (batch_size * no_sentence_per_doc * sen_emb_size)
    '''
    def __init__(self, weights=None):

        super(LSTMSentenceEncoderParallel, self).__init__()
        self.embeddings, vocab, emb_len = create_embedding_layer(weights, config.word_emb_size)
        self.sentenceEncoder = nn.LSTM(config.word_emb_size, 
                                       config.sen_emb_size, 
                                       batch_first=True, 
                                       bidirectional=config.sen_bidirectional)
        self.sen_emb_size = config.sen_emb_size
        if config.sen_bidirectional: self.sen_emb_size *= 2

    def forward(self, input, lengths, sen_len=config.sen_len):
        words = self.embeddings(input.view(-1)).view(-1, sen_len, config.word_emb_size)
        lengths_sorted, idx = torch.sort()
        hn = self.sentenceEncoder(words)[1][0]
        sentences = torch.cat((hn[0], hn[1]), dim=1).reshape(config.batch_size, -1, self.sen_emb_size)
        return sentences


# In[62]:

a = nn.LSTM(10, 12, batch_first=True)


# In[79]:

input = torch.Tensor(np.random.rand(20, 13, 10))


# In[80]:

lengths, idx = torch.sort(torch.LongTensor(np.random.randint(low=1, high=12, size=(20))).reshape(-1), descending=True)


# In[81]:

input_packed = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)


# In[82]:

input_padded = nn.utils.rnn.pad_packed_sequence(input_packed)


# In[83]:

output = nn.utils.rnn.pad_packed_sequence(a(input_packed)[0])


# In[84]:

output[0].shape


# In[ ]:

class SourceBiasParallel(nn.Module):
    '''
    This method is parallel but maynot be as expressive as SourceBiasSeq
    
    Transforms each sentence according to the source its cited from. 
    If a sentence has no such citations, default transformation is used.
    '''
    def __init__(self, no_urls, non_linearity=torch.tanh):
        super(SourceBiasParallel, self).__init__()
        self.trans = nn.Linear(config.sen_emb_size * 2, config.sen_emb_size * 2)
        self.source_embeddings = nn.Embedding(no_urls, config.sen_emb_size * 2)
        self.non_linearity = non_linearity
        
    def forward(self, input, urls):
        sentences = input.reshape(-1, input.size(2))
        urls = self.source_embeddings(urls.reshape(-1))
        
        output = self.trans(sentences)
        output *= urls
        
        return self.non_linearity(output).reshape(input.size())


# In[ ]:

class SourceBiasSeq(nn.Module):
    '''
    Forward prop happens sequentially
    
    Transforms each sentence according to the source its cited from. 
    If a sentence has no such citations, default transformation is used.
    '''
    def __init__(self, no_urls, non_linearity=torch.tanh):
        super(SourceBiasSeq, self).__init__()
        self.trans = nn.Parameter(torch.FloatTensor(random_(no_urls, config.sen_emb_size * 2, config.sen_emb_size * 2)))
        self.bias = nn.Parameter(torch.FloatTensor(random_(no_urls, config.sen_emb_size * 2)))
        self.non_linearity = non_linearity
        
    def forward(self, input, urls):
        sentences = input.reshape(-1, input.size(2))
        urls = urls.reshape(-1)
        
        output = []
        for sen, url in zip(sentences, urls):
            output.append(torch.matmul(sen, self.trans[url]) + self.bias[url])
        output = torch.stack(output, 0)
        
        return self.non_linearity(output).reshape(input.size())


# In[ ]:

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.trans = nn.Bilinear(config.doc_emb_size * 2, config.title_emb_size * 2, 1)
        
    def forward(self, input, context):
        original_shape = input.size()
        context = context.repeat(1, input.size(1)).reshape(-1, context.size(1))
        input = input.reshape(-1, input.size(2))
        
        attention_weights = f.softmax(self.trans(input, context).reshape(original_shape[0:2]).unsqueeze(2), dim=1)
        output = torch.sum(input.reshape(original_shape) * attention_weights, dim=1)
        return output


# In[ ]:

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, int(input_size/2))
        self.layer2 = nn.Linear(int(input_size/2), output_size)
        self.softmax = (output_size > 1)
        
    def forward(self, input):
        h1 = torch.tanh(self.layer1(input))
        h2 = torch.tanh(self.layer2(h1))
        if self.softmax:
            output = f.softmax(h2, dim=1)
        else:
            output = torch.sigmoid(h2)
            
        return output


# In[ ]:

class Model(nn.Module):
    def __init__(self, no_urls, weights=None, use_source_bias=True):
        super(Model, self).__init__()
        self.sentenceEncoder = LSTMSentenceEncoderParallel(weights)
        if use_source_bias:
            self.sourceBias = SourceBiasParallel(no_urls)
        self.use_source_bias = use_source_bias
        self.documentEncoder = nn.LSTM(config.sen_emb_size * 2, config.doc_emb_size, batch_first=True, bidirectional=True)
        self.documentAttention = Attention()
        self.biasMLP = MLP(config.doc_emb_size * 2, 5)
        self.truthMLP = MLP(config.doc_emb_size * 2, 1)
        
    def forward(self, input, urls, titles):
        sentences = self.sentenceEncoder(input)
        if self.use_source_bias:
            bias_sentences = self.sourceBias(sentences, urls)
        else:
            bias_sentences = sentences
        documents = self.documentEncoder(sentences)[0]
        headings = self.sentenceEncoder(titles, config.title_len).squeeze(1)
        document_reps = self.documentAttention(documents, headings)
        bias_output = self.biasMLP(document_reps)
        truth_output = self.truthMLP(document_reps)
        
        return bias_output, truth_output

