from utils.config import *

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



