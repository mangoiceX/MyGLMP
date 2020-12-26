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


class ExternalKnowledge(nn.Module):
    def __init__(self, hop, vocab_size, embedding_dim):

        self.max_hops = hop
        for i in range(self.max_hops + 1):
            C = nn.Embedding(vocab_size, embedding_dim, padding_idx= PAD_token)
            C.weight.data.normal_(0,0.01)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, 'C_')

        self.softmax = nn.Softmax(dim=1)

    def add2memory(self,embed, rnn_output):
        return embed


    def load_memory(self, story, rnn_output, rnn_hidden):  #转载对话历史
        self.m_story = []
        query = rnn_hidden  # 语义转换

        for hop in range(self.max_hops):
            embedding_A = self.C[hop](story)
            embedding_A = torch.sum(embedding_A, 2).squeeze(2)

            if not args['ablationH']:
                embedding_A = self.add2memory(embedding_A, rnn_output)

            prob = (query * embedding_A)
            prob = self.softmax(prob)

            embedding_C = self.C[hop+1](story)
            if not args['ablationH']:
                embedding_C = self.add2memory(embedding_C, rnn_output)

            o_k =  prob * embedding_C
            query = query + o_k






class AttrProxy(object):
    def __init__(self, module, prefix=):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, index):
        return getattr(self.module, self.prefix + str(index))









