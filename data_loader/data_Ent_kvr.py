# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/1/15 15:18
# @File    : data_Ent_kvr.py
"""
文件说明：

"""
from utils.config import *
from utils.utils_general import WordMap, get_data_seq
import json
import ast


def read_file(file_train, entity_map):
    context_arr, conv_arr, kb_info, data = [], [], [], []
    max_resp_len = 0
    with open(file_train, 'r') as f:
        cnt_line = 1
        for line in f:
            line = line.strip()
            if line:
                if '#' in line:
                    line = line.replace('#', '')
                    task_type = line  # 判断是那种任务类型
                    continue

                turn_id, line = line.split(' ', 1)
                if '\t' in line:
                    user_utterance, response, label_ent = line.split('\t')
                    # 利用用户的对话生成四元组(word,$u,turn_id,word_id)
                    gen_u = generate_memory(user_utterance, '$u', turn_id)
                    context_arr += gen_u
                    conv_arr += gen_u

                    label_ent = ast.literal_eval(label_ent)  # 将字符串转化为列表
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather":
                        ent_idx_wet = label_ent
                    elif task_type == "schedule":
                        ent_idx_cal = label_ent
                    elif task_type == "navigate":
                        ent_idx_nav = label_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))  # 直接计算

                    # 计算local pointer的标准值
                    local_ptr = []
                    for key in response.split():
                        for i in range(len(context_arr)-1, -1, -1):
                            if context_arr[i][0] == key:
                                local_ptr.append(i)
                                break
                        else:  # 当没有运行break的时候就运行这里
                            local_ptr.append(len(context_arr))

                    # 计算全局指针，统计上下文有没有在当前回答中出现
                    global_ptr = [1 if (triplet[0] in response.split() or triplet[0] in ent_index) else 0 for triplet in context_arr] +  [
                        1]  # 最后为什么要+1
                    sketch_response = generate_sketch_response(response, entity_map, label_ent, kb_info, task_type)

                    data_details = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # 为什么后面要追加NULL表示符号？
                        'conv_arr': list(conv_arr),  # 要使用list进行转换，否则下面的context_arr改变，这个也会跟着改变
                        'response': response,
                        "local_ptr": local_ptr + [len(context_arr)],
                        'global_ptr': global_ptr,
                        'sketch_response': sketch_response,
                        'ID': cnt_line,
                        'ent_index': ent_index,
                        'ent_idx_wet': ent_idx_wet,
                        'ent_idx_cal': ent_idx_cal,
                        'ent_idx_nav': ent_idx_nav,
                        'kb_info': kb_info
                    }
                    data.append(data_details)
                    gen_r = generate_memory(response, '$r', turn_id)
                    context_arr += gen_r  # 将当前回答加入上下文
                    conv_arr += gen_r
                    max_resp_len = max(max_resp_len, len(response.split()))  # 记录回答的最长长度
                else:  # 表示这段话结束 #这里不知道是做什么
                    response = line
                    kb_item = generate_memory(response, "", turn_id)
                    context_arr = kb_item + context_arr  # conv_arr 没有添加知识
                    kb_info += kb_item
            else:
                cnt_line += 1
                context_arr, conv_arr, kb_info = [], [], []
    return data, max_resp_len


def generate_sketch_response(response, global_entity, label_ent, kb_info, task_type):
    # 对回答的每个单词生成@type的描述
    sketch_response = []
    if not label_ent:
        sketch_response = response.split()
    else:
        for word in response.split():
            if word not in label_ent:
                sketch_response.append(word)
            else:
                entity_type = None
                if task_type != 'weather':
                    for kb_item in kb_info:
                        if word == kb_item[0]:
                            entity_type = kb_item[1]
                            break
                if not entity_type:
                    for key in global_entity:
                        if key != 'poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                entity_type = key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity[key]]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                entity_type = key
                                break
                sketch_response.append('@' + entity_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(tokens, speaker, turn_id):
    tokens = tokens.split(' ')
    gen_content = []
    if speaker == '$u' or speaker == '$r':
        for idx, token in enumerate(tokens):
            tmp = [token, speaker, 'turn' + turn_id, 'word' + str(idx)] + ['PAD'] * (MEM_TOKEN_SIZE - 4)
            gen_content.append(tmp)
    else:
        tokens = tokens[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(tokens))
        gen_content.append(tokens)
    return gen_content


def prepare_data(task, batch_size):
    """
    :param task:
    :return:

    Args:
        batch_size ():
    """
    data_path = '../data/KVR/'
    # 小数据集测试
    # file_train = '{}/dialog-babi-task{}trn-small.txt'.format(data_path, task)
    # file_dev = '{}/dialog-babi-task{}dev-small.txt'.format(data_path, task)
    # file_tst = '{}/dialog-babi-task{}tst-small.txt'.format(data_path, task)
    # file_tst_oov = '{}/dialog-babi-task{}tst-OOV-small.txt'.format(data_path, task)

    # 大数据集测试
    file_train = '{}train.txt'.format(data_path)
    file_dev = '{}dev.txt'.format(data_path)
    file_tst = '{}test.txt'.format(data_path)
    entities_path = '{}kvret_entities.json'.format(data_path)

    with open(entities_path) as f:
        global_entity = json.load(f)

    train_data, max_trn_len = read_file(file_train, global_entity)
    dev_data, max_dev_len = read_file(file_dev, global_entity)
    tst_data, max_tst_len = read_file(file_tst, global_entity)
    max_resp_len = max(max_trn_len, max_tst_len, max_dev_len) + 1

    word_map = WordMap()  # 用来将输入转化为id

    train_loader = get_data_seq(train_data, word_map, batch_size, first=True)
    dev_loader = get_data_seq(dev_data, word_map, batch_size, first=False)
    tst_loader = get_data_seq(tst_data, word_map, batch_size, first=False)

    print("Read %s sentence pairs train" % len(train_loader))
    print("Read %s sentence pairs dev" % len(dev_loader))
    print("Read %s sentence pairs test" % len(tst_loader))
    print("Vocab_size: %s " % word_map.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train_loader, dev_loader, tst_loader, [], word_map, max_resp_len


if __name__ == '__main__':

    prepare_data(1)
