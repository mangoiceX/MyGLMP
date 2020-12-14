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
    context_arr = []
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
                    if key in entityList:
                        for i in range(len(context_arr), -1):
                            if context_arr[i][0] == key:
                                local_ptr.append(i)
                                break
                    else:
                        local_ptr.append(len(context_arr))


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