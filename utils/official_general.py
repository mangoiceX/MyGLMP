import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


class Lang:  # 得到词语与index之间的映射
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, story, trg=False):  # trg判断是否是三元组，因为KB是三元组，而Dialog history是普通形式
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):  # 自定义数据集
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        local_ptr = torch.Tensor(self.data_info['local_ptr'][index])
        global_ptr = torch.Tensor(self.data_info['global_ptr'][index])
        conv_arr = self.data_info['conv_arr'][index]
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id)

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]  # 以字典类型返回当前位置的全部局部变量，也就是将上面的变量一次封装为字典
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):  # 每一个元素就是故事的某个特征，比如context_arr
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)  # 按照故事区分数据
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):  # 这个暂时不懂
        def merge(sequences, story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            if (story_dim):
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i, :end, :] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['conv_arr']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences 
        context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
        response, response_lengths = merge(item_info['response'], False)
        global_ptr, _ = merge_index(item_info['global_ptr'])
        local_ptr, _ = merge(item_info['local_ptr'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
        sketch_response, _ = merge(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)

        # convert to contiguous and cuda
        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        global_ptr = _cuda(global_ptr.contiguous())
        local_ptr = _cuda(local_ptr.contiguous())
        conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        if (len(list(kb_arr.size())) > 1): kb_arr = _cuda(kb_arr.transpose(0, 1).contiguous())

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths

        return data_info  #这里返回的就是批数据

def get_seq(pairs, lang, batch_size, type):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])  # 按照字典里面的键进行聚合
        if (type):  # 因为lang在train数据集上得到单词与id的映射，后面的dev和test使用的也是lang，所以不需要再次hash,改为first
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)

    dataset = Dataset(data_info, lang.word2index, lang.word2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              # shuffle=type,
                                              collate_fn=dataset.collate_fn,
                                              drop_last=True)  # 这个函数处理步骤未理解
    return data_loader
