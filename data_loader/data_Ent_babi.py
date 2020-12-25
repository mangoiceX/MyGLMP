from utils.config import *
from utils.utils_general import WordMap, get_data_seq


def get_type_dict(kb_path):
    with open(kb_path, 'r') as f:
        type_dict = {'R_restaurant': []}
        for line in f.readlines():
            line = line.rstrip('\n')
            line = line.split('\t')
            line[:1] = line[0].split(' ')
            res_name, relationship, entity = line[1], line[2], line[-1]
            if res_name not in type_dict['R_restaurant']:
                type_dict['R_restaurant'].append(res_name)
            if relationship not in type_dict:
                type_dict[relationship] = []
            if entity not in type_dict[relationship]:
                type_dict[relationship].append(entity)
        return type_dict


def get_entity_list(type_dict):
    entity_list = []
    for key in type_dict.keys():
        for item in type_dict[key]:
            entity_list.append(item)
    return entity_list


def read_file(file_train, type_dict, entity_list):
    context_arr, data = [], []
    max_resp_len = 0
    with open(file_train, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                turn_id, line = line.split(' ', 1)
                if '\t' in line:
                    user_utterance, response = line.split('\t')
                    # 利用用户的对话生成四元组(word,$u,turn_id,word_id)
                    gen_u = generate_memory(user_utterance, '$u', turn_id)
                    context_arr += gen_u
                    # 计算local pointer的标准值
                    local_ptr = []
                    ent_word = set()  # 统计response的单词
                    for key in response.split():
                        if key in entity_list:
                            ent_word.add(key)
                            find = False
                            for i in range(len(context_arr) - 1, -1, -1):  # range使用出错
                                if context_arr[i][0] == key:
                                    local_ptr.append(i)
                                    find = True
                                    break
                            if not find:
                                local_ptr.append(len(context_arr))
                        else:
                            local_ptr.append(len(context_arr))
                    # 计算全局指针，统计上下文有没有在当前回答中出现
                    global_ptr = [1 if triplet[0] in response.split() else 0 for triplet in context_arr] +  [
                        1]  # 最后为什么要+1
                    sketch_response = generate_sketch_response(response, entity_list, type_dict)

                    data_details = {
                        'context_arr': context_arr + [['$$$'] * MEM_TOKEN_SIZE],  # 为什么后面要追加NULL表示符号？
                        'response': response,
                        "local_ptr": local_ptr + [len(context_arr)],
                        'global_ptr': global_ptr,
                        'sketch_response': sketch_response
                    }
                    data.append(data_details)
                    gen_r = generate_memory(response, '$r', turn_id)
                    context_arr += gen_r  # 将当前回答加入上下文
                    max_resp_len = max(max_resp_len, len(response.split()))  # 记录回答的最长长度
                else:  # 表示这段话结束 #这里不知道是做什么
                    response = line
                    kb_info = generate_memory(response, "", turn_id)
                    context_arr = kb_info + context_arr
            else:
                context_arr = []

    return data, max_resp_len


def generate_sketch_response(response, global_entity, type_dict):
    # 对回答的每个单词生成@type的描述
    sketch_response = []
    for word in response.split():
        if word in global_entity:
            entity_type = None
            for key in type_dict.keys():
                if word in type_dict[key]:
                    entity_type = key
                    break
            sketch_response.append('@' + entity_type)
        else:
            sketch_response.append(word)
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
        if tokens[1] == "R_rating":
            tokens = tokens + ["PAD"] * (MEM_TOKEN_SIZE - len(tokens))
        else:
            tokens = tokens[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(tokens))
        gen_content.append(tokens)
    return gen_content


def get_seq():
    pass


def prepare_data(task, batch_size):
    """
    :param task:
    :return:

    Args:
        batch_size ():
    """
    data_path = '../data/dialog-bAbI-tasks'
    file_train = '{}/dialog-babi-task{}trn.txt'.format(data_path, task)
    file_dev = '{}/dialog-babi-task{}dev.txt'.format(data_path, task)
    file_tst = '{}/dialog-babi-task{}tst.txt'.format(data_path, task)
    kb_path = '{}/dialog-babi-kb-all.txt'.format(data_path)
    file_tst_oov = '{}/dialog-babi-task{}tst-OOV.txt'.format(data_path, task)

    type_dict = get_type_dict(kb_path)  # 中间结果
    # 实体列表
    entity_list = get_entity_list(type_dict)

    train_data, max_trn_len = read_file(file_train, type_dict, entity_list)
    dev_data, max_dev_len = read_file(file_dev, type_dict, entity_list)
    tst_data, max_tst_len = read_file(file_tst, type_dict, entity_list)
    tst_oov_data, max_tst_oov_len = read_file(file_tst_oov, type_dict, entity_list)
    max_resp_len = max(max_trn_len, max_tst_len, max_dev_len, max_tst_oov_len) + 1

    word_map = WordMap()  # 用来将输入转化为id

    train_loader = get_data_seq(train_data, word_map, batch_size, first=True)
    dev_loader = get_data_seq(dev_data, word_map, batch_size, first=False)
    tst_loader = get_data_seq(tst_data, word_map, batch_size, first=False)
    tst_oov_loader = get_data_seq(tst_oov_data, word_map, batch_size, first=False)

    return train_loader, dev_loader, tst_loader, tst_oov_loader, max_resp_len

if __name__ == '__main__':

    prepare_data(1)
