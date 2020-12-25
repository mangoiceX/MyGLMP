# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2020/12/25 21:17
# @File    : modules.py
"""
文件说明：

"""
import torch.nn as nn
from utils.config import *
import torch
from utils.utils_general import _cuda

class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        self.embedding = nn.Embedding(self.input_size, self.hidden_size, padding_idx = PAD_token)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers, dropout = self.dropout, bidirectional = True)
        self.W = nn.Linear(2*self.hidden_size, self.hidden_size)

    def forward(self, input_seqs):
        embeddings = self.embedding(input_seqs)   # [batch_size, story_length, MEM_TOKEN_SIZE, hidden_size]
        embeddings = torch.sum(embeddings, 2)  # [batch_size, story_length, hidden_size]
        embeddings = self.dropout_layer(embeddings)

        hidden_init = _cuda(torch.zeros(2, input_seqs.size(0), self.hidden_size))  # 隐含状态的初始值

        output, hidden = self.gru(embeddings, hidden_init)

        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1))
        output = self.W(output)

        return output, hidden








