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
from torch import optim, Tensor
from utils.masked_cross_entropy import  masked_cross_entropy
from utils.measures import moses_multi_bleu
from tqdm import tqdm
import numpy as np
import random

class GLMP(nn.Module):
    def __init__(self,hidden_size, word_map, max_resp_len, task, lr, n_layers, dropout, path=None):
        super(GLMP, self).__init__()
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
        self.criterion_bce = nn.BCELoss()  # 这里需要带括号

        self.loss, self.loss_g, self.loss_v, self.loss_l, self.print_every = 0, 0, 0, 0, 1

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
            self.ext_know = ExternalKnowledge(self.max_hops, self.word_map.n_words, self.hidden_size, self.dropout)
            self.decoder = LocalMemory(self.encoder.embedding, self.max_hops, self.word_map, self.hidden_size, self.dropout)

        # 初始化优化器
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.ext_know_optimizer = optim.Adam(self.ext_know.parameters(), lr=self.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)

        # 学习率调控
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)

        # 重置损失参数
        if USE_CUDA:
            self.encoder.cuda()
            self.ext_know.cuda()
            self.decoder.cuda()

    def save_model(self, metrics):  # 模型的存储
        name_data = "KVR/" if self.task == "" else 'BABI/'
        directory = 'save/GLMP-' + name_data + str(self.task) + 'HDD' + str(self.hidden_size) + 'BSZ' + str(args['batch_size']) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + 'LR' + str(self.lr) + 'ACC' + str(metrics)
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.encoder, directory + '/enc.pth')
        torch.save(self.ext_know, directory + '/ext_know.pth')
        torch.save(self.decoder, directory + '/dec.pth')

    def train_batch(self, data, grad_threshold, reset = False):
        if reset:
            self.reset()

        # Zero gradients of all optimizers
        self.encoder_optimizer.zero_grad()
        self.ext_know_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encode and Decode
        max_target_length = self.max_resp_len  # decoder要根据最长长度来生成回答
        all_decoder_output_vocab, local_ptr, _, _, global_ptr = self.encode_and_decode(data, max_target_length, evaluating=False)
        #all_decoder_output_vocab [10, 8, 116]  local_ptr [10 8 70] global_ptr [8 70]
        # Loss calculation and backpropagation
        loss_g = self.criterion_bce(global_ptr, data['global_ptr'])  # 维度一致
        loss_v = masked_cross_entropy(
            all_decoder_output_vocab.transpose(0, 1).contiguous(),  # 前两个维度转化为[batch_size, max_target_length]
            data['sketch_response'].contiguous(),  # [batch_size, max_target_length]
            data['response_lengths']
        )
        loss_l = masked_cross_entropy(
            local_ptr.transpose(0, 1).contiguous(),
            data['local_ptr'].contiguous(),  # [batch_size, max_target_length]
            data['response_lengths']  # 这个没在data中添加
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

    def encode_and_decode(self, data, max_target_length, evaluating=False):
        # 暂时没有添加mask的带啊

        story = data['context_arr']  # [8, 70, 4] 8是bsz, 70是一段对话的单词个数,4是每个词表示的维度
        # encoder_output [batch_size, story_length, hidden_size]  encoder_hidden [batch_size, hidden_size]
        encoder_output, encoder_hidden = self.encoder.forward(data['conv_arr'], data['conv_arr_lengths'])
        # ek_readout [batch_size, hidden_size] global_ptr [batch_size, story_length]
        global_ptr, ek_readout = self.ext_know.load_memory(story, data['conv_arr_lengths'], encoder_output, encoder_hidden)  # ek_readout是q k+1
        sketch_init = torch.cat((encoder_hidden, ek_readout), dim=1)  # 连接hidden_size维度，然后通过一个全连接层降维

        # 通过四元组得到原始对话的单词列表
        copy_list = []
        for each_context in data['context_arr_plain']:
            context = [word_triple[0] for word_triple in each_context]
            copy_list.append(context)
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        all_decoder_output_vocab, all_decoder_output_ptr, decoded_fine, decoded_coarse = self.decoder.forward(
            story.size(),
            self.ext_know,
            global_ptr,
            data['context_arr_lengths'],
            max_target_length,
            args['batch_size'],
            sketch_init,  #
            evaluating,
            copy_list,
            use_teacher_forcing,
            data['sketch_response']
        )

        return all_decoder_output_vocab, all_decoder_output_ptr, decoded_fine, decoded_coarse, global_ptr

    def evaluate(self, dev, metric_best, early_stop = None):
        print('\nSTARTING EVALUATING...')
        self.encoder.train(False)
        self.ext_know.train(False)
        self.decoder.train(False)

        label, pred = [],[]
        acc, total = 0, 0

        for i, data_item in tqdm(enumerate(dev), total=len(dev)):
            _, _, decoded_fine, decoded_coarse, global_ptr = self.encode_and_decode(data_item, self.max_resp_len,
                                                                                    evaluating=True)
            # decoded_fine是以一个batch的一个单词组成的列表为最内维度，所以倒置转化成行为一个完整的句子的预测疏输出
            decoded_fine, decoded_coarse = map(lambda x : np.transpose(np.array(x)), (decoded_fine, decoded_coarse))
            for bi, word_fine in enumerate(decoded_fine):
                response_fine = ''
                for e in word_fine:
                    if e == 'EOS':
                        break
                    response_fine += (e + ' ')
                pred_sentence = response_fine.strip()
                pred.append(pred_sentence)
                label_sentence = data_item['response_plain'][bi].strip()  # 有一次bi会越界
                label.append(label_sentence)

                if pred_sentence == label_sentence:
                    acc += 1
                total += 1
        acc_score = acc / float(total)
        print('ACC SCORE:\t{}'.format(acc_score))

        self.encoder.train(True)
        self.ext_know.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(pred), np.array(label), lowercase=True)  # 暂时无法使用

        if early_stop == 'BLEU':
            if bleu_score >= metric_best:
                self.save_model('BLEU-'+str(bleu_score))
                print('MODEL SAVED')
                return bleu_score
        else:
            if acc_score >= metric_best:
                self.save_model('ACC-'+str(acc_score))
                print('MODEL SAVED')
                return acc_score

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_l = self.loss_l / self.print_every
        print_loss_v = self.loss_v / self.print_every

        self.print_every += 1

        return 'L:{:.2f},LG:{:.2f},LL:{:.2f},LV:{:.2f}'.format(print_loss_avg, print_loss_g, print_loss_l, print_loss_v)








