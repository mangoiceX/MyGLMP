# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2020/12/25 14:52
# @File    : trainer.py
"""
文件说明：

"""


from utils.config import *
from data_loader.data_Ent_babi import prepare_data

train_loader, dev_loader, tst_loader, tst_oov_loader, max_resp_len = prepare_data(args['task'], args['batch_size'])

for i, data in enumerate(train_loader):
    print(data)