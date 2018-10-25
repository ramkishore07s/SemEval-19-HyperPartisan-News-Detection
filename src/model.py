from io import *
import unicodedata
import string
import re
import random
import math
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fcc_matmul(repr_vec, weight, bias, nonlinearity='sigmoid'):
    s = torch.addmv(bias, weight, repr_vec)
    if nonlinearity == 'sigmoid':
        s = torch.sigmoid(s)
    return s

# essential functions for performing the 
def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


class WordEncoderRNN(nn.Module): # Encoder GRU which give hidden state representations
    def __init__(self, batch_size, hidden_size, embed_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_embed_size = embed_size
        self.embedding = nn.Embedding(batch_size, input_embed_size)
        self.gru = nn.GRU(input_embed_size, hidden_size,n_layers, bidirectional=True)

    def forward(self, src, hidden):
        embedded = self.embedding(src)
        outputs, _ = self.gru(embedded, hidden)
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs


class WordAttention(nn.Module): # Attention mechanism for word hidden states
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_W_word = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.bias_word = nn.Parameter(torch.Tensor(hidden_size*2,1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(hidden_size*2, 1))
        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)

    def forward(self, encoder_outputs):
        # encoder_outputs = encoder_outputs.transpose(0, 1) # [B*T*H]
        word_squish = batch_matmul_bias(encoder_outputs, self.weight_W_word,self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1,0))
        word_attn_vectors = attention_mul(encoder_outputs, word_attn_norm.transpose(1,0))
        return word_attn_vectors, word_attn_norm


class SentEncoderRNN(nn.Module): # Encoder GRU which give hidden state representations
    def __init__(self, batch_size, sent_hidden_size, word_hidden_size):
        super().__init__()
        self.sent_hidden_size = sent_hidden_size
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.n_classes = n_classes
        self.gru = nn.GRU(2 * word_hidden_size, 2 * sent_hidden_size,bidirectional=True)
         
    def forward(self, word_attn_vectors, state_sent):
        outputs, _ = self.gru(word_attn_vectors, state_sent)
        outputs = (outputs[:, :, :self.sent_hidden_size] +
                   outputs[:, :, self.sent_hidden_size:])
        return outputs


class SentAttention(nn.Module):
    def __init__(self, sent_gru_hidden):  
        super().__init__()
        self.weight_W_sent = nn.Parameter(torch.Tensor(2*sent_gru_hidden ,2*sent_gru_hidden))
        self.bias_sent = nn.Parameter(torch.Tensor(2*sent_gru_hidden,1))
        self.weight_proj_sent = nn.Parameter(torch.Tensor(2*sent_gru_hidden, 1))
        self.softmax_sent = nn.Softmax()
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1,0.1)
        self.sent_hidden_size = sent_hidden_size
     
    def forward(self, encoder_outputs):      
        sent_squish = batch_matmul_bias(encoder_outputs, self.weight_W_sent,self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn.transpose(1,0))
        sent_attn_vectors = attention_mul(encoder_outputs, sent_attn_norm.transpose(1,0))   
        return sent_attn_vectors, sent_attn_norm 


class HeadlineEncoderRNN(nn.Module):
    def __init__(self, batch_size, hline_hidden_size, sent_hidden_size):
        super().__init__()
        self.hline_hidden_size = hline_hidden_size
        self.batch_size = batch_size
        self.sent_hidden_size = sent_hidden_size
        self.n_classes = n_classes
        self.gru = nn.GRU(2 * sent_hidden_size, hline_hidden_size, bidirectional=True)

    def forward(self, sent_attn_vectors, state_hline):
        outputs, _ = self.gru(sent_attn_vectors, state_hline)
        outputs = (outputs[:, :, :self.hline_hidden_size] +
                   outputs[:, :, self.hline_hidden_size:])
        return outputs 


class HlineAtention(nn.Module):
    def __init__(self, hline_hidden_size):
        super().__init__()
        self.weight_W_hline = nn.Parameter(torch.Tensor(2*hline_hidden_size, 2*hline_hidden_size))
        self.bias_hline = nn.Parameter(torch.Tensor(2*hline_hidden_size, 1))
        self.weight_proj_hline = nn.Parameter(torch.Tensor(2*hline_hidden_size,1))
        self.softmax_hline = nn.Softmax()
        self.weight_W_hline.data.uniform(-0.1, 0.1)
        self.weight_proj_hline.data.uniform(-0.1, 0.1)
        self.hline_hidden_size = hline_hidden_size

        # multi label classifier params
        self.weight_W_fcc = nn.Parameter(torch.Tensor(4, 2*hline_hidden_size))
        self.bias_fcc = nn.Parameter(torch.Tensor(4, 1))
         
    def forward(self, hline_attn_vectors, state_hline):
        hline_squish = batch_matmul_bias(encoder_outputs, self.weight_W_hline,self.bias_hline, nonlinearity='tanh')
        hline_attn = batch_matmul(hline_squish, self.weight_proj_hline)
        hline_attn_norm = self.softmax_hline(hline_attn.transpose(1,0))
        hline_attn_vectors = attention_mul(encoder_outputs, hline_attn_norm.transpose(1,0))  

        # final multi-label prediction
        final_label_vector = fcc_matmul(hline_attn_vectors, self.bias_fcc, self.weight_W_fcc)
        return hline_attn_vectors, final_label_vector


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder    
        
    def forward(self, src, tag, teacher_forcing_tario=0.5):
        pass