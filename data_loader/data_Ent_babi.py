

def get_type_dict(kb_path):
    with open(kb_path,'r') as f:
        type_dict = {}
        type_dict['R_restaurant'] = []
        for line in f.readlines():
            line = line.rstrip('\n')
            line = line.split('\t')
            line[:1] = line[0].split(' ')
            print(line)
            if line[1] not in type_dict['R_restaurant']:
                type_dict['R_restaurant'].append(line[1])
            if line[2] not in type_dict:
                type_dict[line[2]] = []
            if line[-1] not in type_dict[line[2]]:
                type_dict[line[2]].append(line[-1])
        return type_dict

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
    # print(type_dict.keys())

prepare_data(1)