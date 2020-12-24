from utils.config import *
import torch.utils.data as data
import copy
import torch

class WordMap:
    def __init__(self):
        self.word2index = {'UNK': UNK_token, 'PAD': PAD_token, 'SOS': SOS_token, 'EOS': EOS_token}
        self.index2word = dict([(v , k) for k,v in self.word2index.items()])
        self.n_words = len(self.word2index)

    def index_words(self,words):
        try:
            for word_triple in words:
                if type(word_triple) is list:
                    for word in word_triple:
                        self.index_word(word)
                else:
                    self.index_word(word_triple)
        except Exception as e:
            print(e)

    def index_word(self,word):
        if word not in self.word2index.keys():
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    #自定义数据集
    def __init__(self, data_seq, word2id):
        self.data_seq = copy.deepcopy(data_seq)
        self.word2id = word2id
        self.total_len = len(data_seq['context_arr'])

    def __getitem__(self, index):
        context_arr = self.data_seq['context_arr'][index]
        context_arr = self.change_word2id(context_arr,True)

        response = self.data_seq['response'][index]
        response = self.change_word2id(response,False)

        local_ptr = self.data_seq['local_ptr'][index]
        global_ptr = self.data_seq['global_ptr'][index]

        sketch_response = self.data_seq['sketch_response'][index]
        sketch_response = self.change_word2id(sketch_response,False)

        data_info = {}
        for key in self.data_seq.keys():
            try:
                data_info[key] = locals()[key]
            except Exception as e:
                print("locals() failed")

        return data_info

    def __len__(self):
        return self.total_len



    def change_word2id(self,data,isTriple = False):
        if isTriple:
            ids = [self.word2id[word] if word in self.word2id else UNK_token for word in data.split(',') ] + [EOS_token]
        else:
            ids = []
            for i,word_triplet in enumerate(data):
                ids.append([])
                for word in word_triplet:
                    tmp = self.word2id[word] if word in self.word2id else UNK_token
                    ids[i].append(tmp)
        ids = torch.Tensor(ids)

        return ids



def get_data_seq(data,word_map,first):
    data_seq = {}
    for key in data[0].keys():
        data_seq[key] = []

    for data_item in data:
        for key in data_item.keys():
            data_seq[key].append(data_item[key])
        if first:
            word_map.index_words(data_item['context_arr'])
            word_map.index_words(data_item['response'])
            word_map.index_words(data_item['sketch_response'])
    print(data_seq)
    print('*'*50)
    print(word_map.word2index)
    # 制作数据集
    dataset = Dataset(data_seq,word_map.word2index)
    #制作批量训练数据






