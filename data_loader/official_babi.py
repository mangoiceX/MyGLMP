import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast

from utils.official_general  import *
from utils.utils_temp import entityList, get_type_dict


def read_langs(file_name, global_entity, type_dict, max_line=None):
    # print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    max_resp_len, sample_counter = 0, 0
    with open(file_name) as fin:
        cnt_lin = 1
        for line in fin:
            line = line.strip()
            if line:
                nid, line = line.split(' ', 1)
                # print("line", line)
                if '\t' in line:
                    u, r = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u
                    local_ptr, ent_words = [], []

                    # Get local pointer position for each word in system response
                    for key in r.split():
                        if key in global_entity and key not in ent_words:
                            ent_words.append(key)  # 收集回答中在系统词汇中的单词
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in global_entity)]
                        # 统计回答中的单词在总的实体列表中的最后位置
                        index = max(index) if (index) else len(context_arr)
                        local_ptr.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    global_ptr = [1 if (word_arr[0] in ent_words or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]
                    # 如果用户提问的单词没在回答中，那么为0，否则为1

                    sketch_response = generate_template(global_entity, r, type_dict)

                    data_detail = {  # 一段对话的所有信息
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        # 形式[[word,$u,turn_id,word_id],...]
                        'response': r,
                        'sketch_response': sketch_response,
                        'local_ptr': local_ptr + [len(context_arr)],  # local pointer的标签值，统计回答中的单词在总的实体列表中的最后位置
                        'global_ptr': global_ptr,  # global pointer的标签值，如果用户提问的单词没在回答中，那么为0，否则为1
                        'ent_index': ent_words,
                        'ent_idx_cal': [],
                        'ent_idx_nav': [],
                        'ent_idx_wet': [],
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': ""}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))  # 将response做成(word,speaker,turn_idx,word_idx)形式
                    context_arr += gen_r
                    conv_arr += gen_r
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line
                    kb_info = generate_memory(r, "", str(nid))  # 这个是用来干嘛的
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []  # 一段对话结束，清空。对话与对话之间有空行分开
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len  # 数据和最大的回答长度


def generate_memory(sent, speaker, time):  # 生成memory的知识
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":  # 将对话格式化，存入memory中
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        if sent_token[1] == "R_rating":  # 记录饭店的星级，是一个简单的数字
            sent_token = sent_token + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        else:  # kb-all都是R开头，表示是关系，因为R_rating是数字，所以单独处理
            sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def generate_template(global_entity, sentence, type_dict):  # sketch-RNN生成sketch_response，
    sketch_response = []
    for word in sentence.split():
        if word in global_entity:
            ent_type = None
            for kb_item in type_dict.keys():
                if word in type_dict[kb_item]:
                    ent_type = kb_item
                    break
            sketch_response.append('@' + ent_type)  # 通过词找到他属于的关系，比如根据3 miles找到他所在的关系distance
        else:
            sketch_response.append(word)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def prepare_data_seq(task, batch_size):
    data_path = '../data/dialog-bAbI-tasks/dialog-babi'
    file_train = '{}-task{}trn-small.txt'.format(data_path, task)
    file_dev = '{}-task{}dev-small.txt'.format(data_path, task)
    file_test = '{}-task{}tst-small.txt'.format(data_path, task)
    kb_path = data_path + '-kb-all.txt'
    # print(kb_path)
    file_test_OOV = '{}-task{}tst-OOV-small.txt'.format(data_path, task)  # OOV文件是out of vocabulary测试文件
    type_dict = get_type_dict(kb_path, dstc2=False)  # 三元组词典，{饭店和关系作为键，其他作为值}饭店都是Subject,其他都是Object
    global_ent = entityList('../data/dialog-bAbI-tasks/dialog-babi-kb-all.txt', int(task))  # 收集所有的Subject和Object

    pair_train, train_max_len = read_langs(file_train, global_ent, type_dict)
    pair_dev, dev_max_len = read_langs(file_dev, global_ent, type_dict)
    pair_test, test_max_len = read_langs(file_test, global_ent, type_dict)
    pair_testoov, testoov_max_len = read_langs(file_test_OOV, global_ent, type_dict)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len, testoov_max_len) + 1

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    # 最后一个参数的解释：因为lang在train数据集上需要做进行编码，后面的dev和test使用的也是lang，所以不需要再次hash,改为first
    dev = get_seq(pair_dev, lang, batch_size, False)  # 之前batch_size固定为100
    test = get_seq(pair_test, lang, batch_size, False)
    testoov = get_seq(pair_testoov, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, testoov, lang, max_resp_len


def get_data_seq(file_name, lang, max_len, task=5, batch_size=1):  # 代碼中並沒有用到
    data_path = 'data/dialog-bAbI-tasks/dialog-babi'
    kb_path = data_path + '-kb-all.txt'
    type_dict = get_type_dict(kb_path, dstc2=False)
    global_ent = entityList(kb_path, int(task))
    pair, _ = read_langs(file_name, global_ent, type_dict)
    # print("pair", pair)
    d = get_seq(pair, lang, batch_size, False)
    return d
