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
        super().__init__()
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
    def __init__(self, max_hops, vocab_size, embedding_dim):
        super().__init__()
        self.max_hops = max_hops
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab_size, embedding_dim, padding_idx= PAD_token)
            C.weight.data.normal_(0,0.01)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, 'C_')

        self.softmax = nn.Softmax(dim=1)

    def add2memory(self, embed, rnn_output):
        return embed

    def load_memory(self, story, rnn_output, rnn_hidden):  #转载对话历史
        self.m_story = []
        query = rnn_hidden  # 语义转换

        for hop in range(self.max_hops):
            embedding_A = self.C[hop](story)
            embedding_A = torch.sum(embedding_A, 2).squeeze(2)

            if not args['ablationH']:
                embedding_A = self.add2memory(embedding_A, rnn_output)

            prob_origin = (query * embedding_A)  # 查询向量和内存中的内容相乘计算出注意力分布
            prob = self.softmax(prob_origin)

            embedding_C = self.C[hop+1](story)
            if not args['ablationH']:
                embedding_C = self.add2memory(embedding_C, rnn_output)

            o_k = prob * embedding_C  # 注意力分布和下一跳的存储内容相乘得到此时的输出
            query = query + o_k  # 第k+1层的输入等于第k层的输入与输出相加

            self.m_story.append(embedding_A)  # 保存第k个过程的矩阵
        self.m_story.append(embedding_C)
        return nn.sigmoid(prob_origin), query

    def forward(self, rnn_hidden, global_ptr):
        query = rnn_hidden

        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if not args['ablationG']:
                m_A =  m_A * global_ptr

            prob_origin = query * m_A
            prob_soft = self.softmax(prob_origin)

            m_C = self.m_story[hop+1]
            if not args['ablationG']:
                m_C = m_C * global_ptr
            o_k = prob_soft * m_C
            query = query + o_k

        return prob_origin, prob_soft  # 注意力分布 输出分布


class LocalMemory(nn.Module):
    def __init__(self, shared_embed, max_hops, word_map, embedding_dim, dropout):
        super().__init__()
        self.max_hops = max_hops
        self.softmax = nn.Softmax(dim=1)
        self.word_map = word_map
        self.num_vocab = word_map.n_words
        self.sketch_rnn = nn.GRU(embedding_dim,embedding_dim,dropout=dropout)
        self.projector = nn.Linear(2*embedding_dim, embedding_dim)
        self.C = shared_embed

    def forward(self, story, extKnow, global_ptr, story_length, max_target_length, batch_size, encoded_hidden, evaluating, copy_list):
        record = _cuda(torch.ones(story.size(0), story.size(1)))
        all_decoder_output_ptr = _cuda(torch.zeros(max_target_length, batch_size, story.size(1)))  # 这个初始化维度如何确定的
        all_decoder_output_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        sketch_response = _cuda(torch.LongTensor([SOS_token] * batch_size))  # 为什么是batch_size
        hidden_init = self.projector(encoded_hidden)
        decoded_fine, decoded_coarse = [], []

        # 使用sketch RNN逐字生成输出
        for t in range(max_target_length):
            _, hidden = self.sketch_rnn(sketch_response, hidden_init)
            query = hidden[0]

            p_vocab = self.softmax(self.C.weight * hidden)
            _, top_p_vocab = p_vocab.data.topk(1)
            all_decoder_output_vocab[t] = p_vocab

            # 使用sketch rnn的最后隐含态查询EK得到注意力分布，也就是local pointer
            local_ptr, prob_soft = extKnow(query, global_ptr)
            all_decoder_output_ptr[t] = local_ptr

            if evaluating:
                search_len = min(5, min(story_length))
                prob_soft = prob_soft * record
                _, top_p_soft = prob_soft.data.topk(search_len)

                tmp_f, tmp_c = [], []
                for bi in range(batch_size):
                    token = top_p_vocab[bi].item()
                    tmp_c.append(self.word_map.index2word[token])

                    if '@' in self.word_map.index2word[token]:  #'@R_cuisine','@R_location','@R_number','@R_price'
                        for i in range(search_len):
                            cw = 'UNK'
                            if top_p_soft[i][bi] < story_length[bi]-1:
                                cw = copy_list[bi][top_p_soft[i][bi].item()]
                                break
                            tmp_f.append(cw)
                        if args['record']:
                            record[bi][top_p_soft[i][bi].item()] = 0  # copy_list中已经使用的部分清零
                    else:
                        tmp_f.append(token)  # 如果不是那几个‘@’的话，则记录单词
                decoded_fine.append(tmp_f)
                decoded_coarse.append(tmp_c)

        return all_decoder_output_vocab, all_decoder_output_ptr, decoded_fine, decoded_coarse


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, index):
        return getattr(self.module, self.prefix + str(index))









