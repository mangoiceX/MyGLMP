# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2020/12/30 11:24
# @File    : GLMP.py
"""
文件说明：将之前的组件组装成一个完成的模型

"""
import torch.nn as nn
import torch
from utils.config import *
from models.modules import ContextRNN, ExternalKnowledge, LocalMemory
from torch import optim


class GLMP(nn.Module):
    def __init__(self,hidden_size, word_map, max_resp_len, task, lr, n_layers, dropout, path=None):
        super().__init__()
        self.name = "GLMP"
        self.input_size = word_map.n_words
        self.hidden_size = hidden_size
        self.output_size = word_map.n_words
        self.word_map = word_map
        self.max_resp_len = max_resp_len
        self.task = task
        self.lr = lr
        self.max_hops = n_layers  # decoder的层数就是hop的数量
        self.n_layers = n_layers
        self.dropout = dropout

        #  初始化基本组件
        if path:  # 默认加载的参数是CPU参数
            if USE_CUDA:
                print('Model {} loaded'.format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.pth', lambda storage, loc : storage.cuda)
                self.ext_know = torch.load(str(path)+'/ext_know.pth', lambda storage, loc : storage.cuda)
                self.decoder = torch.load(str(path)+'/dec.pth', lambda storage, loc : storage.cuda)
            else:
                print('Model {} loaded'.format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.pth')
                self.ext_know = torch.load(str(path)+'/ext_know.pth')
                self.decoder = torch.load(str(path)+'/dec.pth')
        else:
            self.encoder = ContextRNN(self.input_size, self.hidden_size, self.n_layers, self.dropout)
            self.ext_know = ExternalKnowledge(self.max_hops, self.word_map.n_words, self.hidden_size)
            self.decoder = LocalMemory(self.encoder.embedding, self.max_hops, self.word_map, self.hidden_size, self.dropout)

        # 初始化优化器
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.ext_know_optimizer = optim.Adam(self.ext_know.parameters(), lr=self.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)

        # 学习率调控
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=2, min_lr=0.0001, verbose=True)

        # 重置损失参数
        if USE_CUDA:
            self.encoder.cuda()
            self.ext_know.cuda()
            self.decoder.cuda()

    def save_model(self, acc):  # 模型的存储
        name_data = "KVR/" if self.task == "" else 'BABI/'
        directory ='save/GLMP-' + name_data + str(self.task) + 'HDS' + str(self.hidden_size) + 'BSZ' \
                   + str(args['batch_size']) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + \
                    + 'LR' + str(self.lr) + 'ACC' + str(acc)
        if os.path.exists(directory):
            os.path.mkdir(directory)

        torch.save(self.encoder, directory + '/enc.pth')
        torch.save(self.ext_know, directory + '/ext_know.pth')
        torch.save(self.decoder, directory + '/dec.pth')


    def train_batch(self, data, grad_threshold, reset = False):
        if reset:
            self.rest()

        # Zero gradients of all optimizers
        self.encoder_optimizer.zero_grad()
        self.ext_know_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encode and Decode
        max_target_length = max(data['max_resp_len'])  # decoder要根据最长长度来生成回答
        all_decoder_output_vocab, local_ptr, global_ptr = self.encode_and_decode(data, max_target_length, evaluating = False)

        # Loss calculation and backpropagation
        loss_g = nn.BCELoss(global_ptr, data['global_ptr'])
        loss_v = masked_cross_entropy(
            all_decoder_output_vocab,
            data['sketch_response'],
            data['response_lengths']  # 这个没在data中添加
        )
        loss_l = masked_cross_entropy(
            local_ptr,
            data['local_ptr'],
            data['local_ptr_length']  # 这个没在data中添加
        )

        loss = loss_g + loss_v + loss_l
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), grad_threshold)
        ek = torch.nn.utils.clip_grad_norm_(self.ext_know.parameters(), grad_threshold)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), grad_threshold)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.ext_know_optimizer.step()
        self.decoder_optimizer.step()

        self.loss += loss.item()
        self.loss_g += loss_g.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()

    def reset(self):
        self.loss, self.loss_g, self.loss_v, self.loss_l, self.print_every = 0, 0, 0, 0, 1









