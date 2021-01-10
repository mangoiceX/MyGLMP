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
import copy


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.input_size = input_size  # 116 是词汇表的大小
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        self.embedding = nn.Embedding(self.input_size, self.hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers, dropout=self.dropout, bidirectional=True,
                          batch_first=True)
        #  gru的hidden state的维度是(num_layers*num_directions, batch, hidden size)
        self.W = nn.Linear(2*self.hidden_size, self.hidden_size)

    def forward(self, input_seqs, input_lengths):  # input_lengths是该batch中，每个故事的长度
        #embeddings = self.embedding(input_seqs)   # [batch_size, story_length, MEM_TOKEN_SIZE, hidden_size]
        # 保持batch_size维度不变，另外两个维度合并，然后在embedding
        embeddings = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embeddings = embeddings.view(input_seqs.size() + (embeddings.size(-1),))
        embeddings = torch.sum(embeddings, 2)  # [batch_size, story_length, hidden_size]
        embeddings = self.dropout_layer(embeddings)  # 为什么要使用dropout

        hidden_init = _cuda(torch.zeros(2*self.n_layers, input_seqs.size(1), self.hidden_size))  # [2, batch_size, hidden_size]隐含状态的初始值
        if input_lengths:
            embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, batch_first=False)
        output, hidden = self.gru(embeddings, hidden_init)  # output [] hidden [2, batch_size, hidden_size]
        # outputs   (seq_len, batch, num_directions * hidden_size) hidden [2 8 128](num_layers * num_directions, batch, hidden_size)

        if input_lengths:  # 消除pack_padded_sequence的填充
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  #
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1))
        output = self.W(output)  # [batch_size, story_length, hidden_size]

        return output, hidden


class ExternalKnowledge(nn.Module):
    def __init__(self, max_hops, vocab_size, embedding_dim, dropout):
        super().__init__()
        self.max_hops = max_hops
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, 'C_')

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def add2memory(self, embed, rnn_output, conv_arr_lengths):  # 调试到这里
        for bi in range(embed.size(0)):
            start, end = 0, conv_arr_lengths[bi]
            embed[bi, start:end, :] = embed[bi, start:end, :] + rnn_output[bi, :end, :]
        return embed

    def load_memory(self, story, conv_arr_lengths, rnn_output, rnn_hidden):  #转载对话历史
        self.m_story = []
        query = rnn_hidden  # 语义转换 [batch_size, hidden_size]
        story_size = story.size()

        for hop in range(self.max_hops):
            #embedding_A = self.C[hop](story)
            embedding_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # 要转化为[batch_size, story_length*4]这样之后为什么会效果好些，是embed算法要求这样做的吗
            embedding_A = embedding_A.view(story_size + (embedding_A.size(-1),))  # b * m * s * e
            embedding_A = torch.sum(embedding_A, 2)  # 合并用来表示每个词的维度-4  [batch_size,story_length,hidden_size]
            if not args['ablationH']:
                embedding_A = self.add2memory(embedding_A, rnn_output, conv_arr_lengths)
            embedding_A = self.dropout_layer(embedding_A)  # 为什么要添加dropout

            query_tmp = query.unsqueeze(1).expand_as(embedding_A)  # 需要对第二个维度进行拓展
            prob_origin = torch.sum(embedding_A*query_tmp, 2)  # 查询向量和内存中的内容相乘计算出注意力分布
            prob = self.softmax(prob_origin)  #[batch_size, story_length]

            #embedding_C = self.C[hop+1](story)
            embedding_C = self.C[hop+1](story.contiguous().view(story_size[0], -1).long())
            embedding_C = embedding_C.view(story_size + (embedding_C.size(-1),))  # b * m * s * e
            embedding_C = torch.sum(embedding_C, 2)
            if not args['ablationH']:
                embedding_C = self.add2memory(embedding_C, rnn_output, conv_arr_lengths)

            prob_tmp = prob.unsqueeze(2).expand_as(embedding_C)
            o_k = torch.sum(embedding_C * prob_tmp, 1)  # 注意力分布和下一跳的存储内容相乘得到此时的输出 , 消除story_lenght
            query = query + o_k  # 第k+1层的输入等于第k层的输入与输出相加

            self.m_story.append(embedding_A)  # 保存第k个过程的矩阵
        self.m_story.append(embedding_C)  # 最后一跳的内容A没有保存，C才有
        return self.sigmoid(prob_origin), query  # global pointer，和KB中读出来的值

    def forward(self, rnn_hidden, global_ptr):
        '''

        Args:
            rnn_hidden (): [batch_size, hidden_size]
            global_ptr ():[batch_size, story_length]

        Returns:

        '''

        query = rnn_hidden

        for hop in range(self.max_hops):
            m_A = self.m_story[hop]  # [batch_size, story_length, hidden_size]
            if not args['ablationG']:
                m_A = m_A * global_ptr.unsqueeze(2).expand_as(m_A)
            query_tmp = query.unsqueeze(1).expand_as(m_A)  # 增加story_length维度, 会不会添加的数据有问题
            prob_origin = torch.sum(m_A*query_tmp, 2)  # 消除hidden_size维度
            prob_soft = self.softmax(prob_origin)  # [batch_size, story_length]

            m_C = self.m_story[hop+1]
            if not args['ablationG']:
                m_C = m_C * global_ptr.unsqueeze(2).expand_as(m_C)
            prob_tmp = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob_tmp, 1)  # 消除story_length维度
            query = query + o_k

        return prob_origin, prob_soft  # 注意力分布 输出分布


class LocalMemory(nn.Module):
    def __init__(self, shared_embed, max_hops, word_map, embedding_dim, dropout):
        super().__init__()
        self.max_hops = max_hops
        self.softmax = nn.Softmax(dim=1)
        self.word_map = word_map
        self.dropout_layer = nn.Dropout(dropout)
        self.num_vocab = word_map.n_words
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.projector = nn.Linear(2*embedding_dim, embedding_dim)  # 默认num_layers=1
        self.C = shared_embed
        self.relu = nn.ReLU()

    def forward(self, story_size, ext_know, global_ptr, story_length, max_target_length, batch_size, encoded_hidden,
                evaluating, copy_list):
        record = _cuda(torch.ones(story_size[0], story_size[1]))  # [batch_size, story_length]
        # all_decoder_output_ptr输出的是局部指针，是针对当前对话来说的
        all_decoder_output_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))  # 针对当前对话
        # all_decoder_output_vocab 针对的是词汇表
        all_decoder_output_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))  # 针对词汇表
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))  # 每次为同一个batch的样本生成一个单词
        hidden_init = self.relu(self.projector(encoded_hidden)).unsqueeze(0)  # 对连接降维, 为什么要添加relu
        decoded_fine, decoded_coarse = [], []

        # 使用sketch RNN逐字生成输出
        for t in range(max_target_length):
            sketch_response = self.dropout_layer(self.C(decoder_input))  #[8] -> [1,8,128] .
            if len(sketch_response.size()) == 2:
                sketch_response = sketch_response.unsqueeze(0)
            _, hidden = self.sketch_rnn(sketch_response, hidden_init)  # [seq_len, batch_size, embedding_dim]
            query = hidden[0]  # [num_layers * num_directions, batch, embedding_dim]  我认为结果包含了各层的隐含态
            # p_vocab [batch_size, vocab_size]
            p_vocab = hidden.squeeze(0).matmul(self.C.weight.transpose(1,0))  # 这里添加softmax层导致效果变差
            # p_vocab [vocab_size, embedding_dim]
            all_decoder_output_vocab[t] = p_vocab
            _, top_p_vocab = p_vocab.data.topk(1)  # 获得上下文关注的词汇的序号，这是针对词汇表

            # 使用sketch rnn的最后隐含态查询EK得到注意力分布，也就是local pointer
            local_ptr, prob_soft = ext_know(query, global_ptr)  # 如果不适用forward也可以达到同样效果吗
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
                        cw = 'UNK'  # 改为数值
                        for i in range(search_len):
                            if top_p_soft[bi][i] < story_length[bi]-1:  # top_p_soft[i][bi] -> top_p_soft[:, i][bi]
                                cw = copy_list[bi][top_p_soft[bi][i].item()]
                                break
                        tmp_f.append(cw)  # 这个是放在循环外面
                        if args['record']:
                            record[bi][top_p_soft[bi][i].item()] = 0  # copy_list中已经使用的部分清零
                    else:
                        tmp_f.append(self.word_map.index2word[token])  # 如果不是那几个‘@’的话，则记录单词
                decoded_fine.append(tmp_f)
                decoded_coarse.append(tmp_c)

        return all_decoder_output_vocab, all_decoder_output_ptr, decoded_fine, decoded_coarse


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, index):
        return getattr(self.module, self.prefix + str(index))









