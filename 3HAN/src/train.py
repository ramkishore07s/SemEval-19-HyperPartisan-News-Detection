import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import *
# from utils import load_dataset
import numpy as np

def load_embeddings(file):
    print("Loading glove model.....")
    f = open(file, "r+")
    model = {}
    for line in f:
        s_line = line.split()
        word = s_line[0]
        embedding = np.array([float(val) for val in s_line[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()

if __name__ == "__main__":
    # args = parse_args()
    hidden_size = 512
    embed_size = 300
    # assert torch.cuda.is_available()

    # print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    # seq2seq = Seq2Seq(encoder, decoder).cuda()
    # optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    # print(seq2seq)
    embeddings = load_embeddings("glove.6B/300d.txt")
