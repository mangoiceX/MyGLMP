from utils.config import *
import copy
import torch


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


class WordMap:
    def __init__(self):
        self.word2index = {'UNK': UNK_token, 'PAD': PAD_token, 'SOS': SOS_token, 'EOS': EOS_token}
        self.index2word = dict([(v, k) for k, v in self.word2index.items()])
        self.n_words = len(self.word2index)

    def index_words(self, words, is_triple=True):
        if is_triple:
            for word in words.split(' '):
                self.index_word(word)
        else:
            for word_triple in words:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index.keys():
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(torch.utils.data.Dataset):
    # 自定义数据集
    def __init__(self, data_seq, word2id):
        self.data_seq = copy.deepcopy(data_seq)
        self.word2id = word2id
        self.total_len = len(data_seq)  # 因为我是按照样本数作为第一维度，所以直接统计list长度

    def __getitem__(self, index):  # 当使用索引时，就会调用该函数
        context_arr = self.data_seq[index]['context_arr']
        context_arr = self.change_word2id(context_arr, True)

        conv_arr = self.data_seq[index]['conv_arr']
        conv_arr = self.change_word2id(conv_arr, True)

        response = self.data_seq[index]['response']
        response = self.change_word2id(response, False)

        local_ptr = self.data_seq[index]['local_ptr']
        global_ptr = self.data_seq[index]['global_ptr']

        sketch_response = self.data_seq[index]['sketch_response']
        sketch_response = self.change_word2id(sketch_response, False)

        data_info = {}
        for key in self.data_seq[0].keys():
            try:
                data_info[key] = locals()[key]
            except:
                data_info[key] = self.data_seq[index][key]

        # 保存为编码的数据，模型评估时需要使用
        data_info['context_arr_plain'] = self.data_seq[index]['context_arr']
        data_info['response_plain'] = self.data_seq[index]['response']

        return data_info

    def __len__(self):
        return self.total_len

    def change_word2id(self, data, is_triple=False):
        if not is_triple:
            ids = [self.word2id[word] if word in self.word2id else UNK_token for word in data.split(' ')] + [EOS_token]
        else:
            ids = []
            for i, word_triplet in enumerate(data):
                ids.append([])
                for word in word_triplet:
                    tmp = self.word2id[word] if word in self.word2id else UNK_token
                    ids[i].append(tmp)
        ids = torch.Tensor(ids)

        return ids

    def collate_fn(self, data):  # 默认的collate_fn的输入参数是batch,是将batch_size个__getitem__的返回结果组成batch

        def merge(sequences, is_triple = False, pad_zeros = False):
            lengths = [len(seq) for seq in sequences]
            max_length = max(lengths) if max(lengths) > 0 else 1
            if pad_zeros:
                padded_seqs = torch.zeros(len(sequences), max_length).float()
            else:
                if is_triple:
                    padded_seqs = torch.ones(len(sequences), max_length, MEM_TOKEN_SIZE).long()
                else:
                    padded_seqs = torch.ones(len(sequences), max_length).long()

            for i, seq in enumerate(sequences):
                end = lengths[i]
                seq = torch.Tensor(seq)
                try:
                    padded_seqs[i, :end] = seq[:end]  # seq是[[],[]]的新式，也就是一份data_details的数据
                except:
                    print(padded_seqs.shape)
                    print(seq.shape)

            return padded_seqs, lengths

        data.sort(key=lambda x: len(x['conv_arr']), reverse=True)  # 暂时使用context_arr，而不是conv_arr ,rnn的pack_padded_sequence要求
        item_info = {}
        for key in data[0].keys():  # 按照内容聚合
            item_info[key] = [d[key] for d in data]

        # merge sequences
        context_arr, context_arr_lengths = merge(item_info['context_arr'], is_triple = True)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], is_triple = True)
        response, response_lengths = merge(item_info['response'], is_triple = False)
        sketch_response, sketch_response_lengths = merge(item_info['sketch_response'], is_triple = False)

        # merge id
        global_ptr, global_ptr_lengths = merge(item_info['global_ptr'], is_triple = False, pad_zeros = True)
        local_ptr, local_ptr_lengths = merge(item_info['local_ptr'], is_triple = False, pad_zeros = False)

        # convert to contiguous and cuda
        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        global_ptr = _cuda(global_ptr.contiguous())
        local_ptr = _cuda(local_ptr.contiguous())

        data_info = {}
        for key in item_info.keys():
            try:
                data_info[key] = locals()[key]
            except:
                data_info[key] = item_info[key]

        # additional plain information
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['local_ptr_lengths'] = local_ptr_lengths

        return data_info


def get_data_seq(data, word_map, batch_size, first = False):
    """
    Args:
        batch_size ():
        data (list):list的每个元素都是一个字典
        word_map (WordMap对象): 利用对象保存word与id的映射
        first (bool):是否是第一次统计，只统计一次，完成word与id的映射

    Returns:

    """

    if first:  # 只处理train数据集
        for data_item in data:
            word_map.index_words(data_item['context_arr'], is_triple=False)
            word_map.index_words(data_item['response'], is_triple=True)
            word_map.index_words(data_item['sketch_response'], is_triple=True)

    # 制作数据集
    dataset = Dataset(data, word_map.word2index)
    # 制作批量训练数据
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                  batch_size = batch_size,
                                  # shuffle = first,
                                  collate_fn = dataset.collate_fn,
                                  drop_last=True
                                  )

    return data_loader