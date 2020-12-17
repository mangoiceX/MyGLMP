from utils.config import *

def get_type_dict(kb_path):
    with open(kb_path,'r') as f:
        type_dict = {}
        type_dict['R_restaurant'] = []
        for line in f.readlines():
            line = line.rstrip('\n')
            line = line.split('\t')
            line[:1] = line[0].split(' ')
            res_name,relationship,entity = line[1],line[2],line[-1]
            if res_name not in type_dict['R_restaurant']:
                type_dict['R_restaurant'].append(res_name)
            if relationship not in type_dict:
                type_dict[relationship] = []
            if entity not in type_dict[relationship]:
                type_dict[relationship].append(entity)
        return type_dict


def getEntityList(type_dict):
    entityList = []
    for key in type_dict.keys():
        for item in type_dict[key]:
            entityList.append(item)
    return entityList

def read_file(file_train, type_dict, entityList):
    ent_word = set()  #统计response的单词
    context_arr,data = [],[]
    with open(file_train,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            turn_id, line = line.split(' ',1)
            if '\t' in line:
                user_utterance, response = line.split('\t')
                #利用用户的对话生成四元组(word,$u,turn_id,word_id)
                gen_u = generate_memory(user_utterance,'$u',turn_id)
                context_arr += gen_u
                #计算local pointer的标准值
                local_ptr = []
                for key in response.split(' '):
                    ent_word.add(key)
                    if key in entityList:
                        for i in range(len(context_arr), -1):
                            if context_arr[i][0] == key:
                                local_ptr.append(i)
                                break
                    else:
                        local_ptr.append(len(context_arr))
                #计算全局指针，统计上下文有没有在当前回答中出现
                global_ptr =[1 if triplet[0] in response.split() else 0 for triplet in context_arr]
                sketch_response = generate_sketch_response(response,entityList,type_dict)

                gen_r = generate_memory(response, '$r', turn_id)
                context_arr += gen_r #将当前回答加入上下文
                data_details = {
                    'context_arr':context_arr +[['$$$']*MEM_TOKEN_SIZE], #为什么后面要追加NULL表示符号？
                    'response':response,
                    "local_ptr":local_ptr,
                    'global_ptr':global_ptr,
                    'sketch_response':sketch_response
                }
                data.append(data_details)
            else: #表示这段话结束
                context_arr = []

    return data


def generate_sketch_response(response,global_entity,type_dict):
    #对回答的每个单词生成@type的描述
    sketch_response = []
    for word in response:
        if word in global_entity:
            entity_type = None
            for key in type_dict.keys():
                if word in type_dict[key]:
                    entity_type = key
                    break
            sketch_response.append('@'+entity_type)
        else:
            sketch_response.append(word)
    sketch_response = " ".join(sketch_response)
    return sketch_response

def generate_memory(tokens,speaker,turn_id):
    tokens = tokens.split(' ')
    gen_content = []
    if speaker == '$u' or speaker == '$r':
        for idx, token in enumerate(tokens):
            tmp = [token, speaker, 'turn'+str(turn_id),'word'+str(idx)] + ['PAD']*(MEM_TOKEN_SIZE-4)
            gen_content.append(tmp)

    return gen_content







def prepare_data(task):
    '''
    :param dataset:
    :return:
    '''
    data_path = '../data/dialog-bAbI-tasks'
    file_train = '{}/dialog-babi-task{}trn.txt'.format(data_path,task)
    file_dev = '{}/dialog-babi-task{}dev.txt'.format(data_path,task)
    file_tst = '{}/dialog-babi-task{}tst.txt'.format(data_path,task)
    kb_path = '{}/dialog-babi-kb-all.txt'.format(data_path)
    file_tst_oov = '{}/dialog-babi-task{}tst-OOV.txt'.format(data_path,task)

    type_dict = get_type_dict(kb_path)    #中间结果
    #实体列表
    entityList = getEntityList(type_dict)

    train_data = read_file(file_train, type_dict, entityList)
    # print(type_dict.keys())
prepare_data(1)