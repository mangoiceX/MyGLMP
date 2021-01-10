# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2020/12/25 14:52
# @File    : trainer.py
"""
文件说明：

"""


from utils.config import *
from data_loader.official_babi import prepare_data_seq  # data_Ent_babi
from data_loader.data_Ent_babi import prepare_data
from models.GLMP import GLMP
from tqdm import tqdm
import math

metrics_best, cnt = 0.0, 0

# 转换数据
train_loader, dev_loader, tst_loader, tst_oov_loader, word_map, max_resp_len = prepare_data_seq(args['task'],
                                                                                            args['batch_size'])
model = GLMP(args['hidden_size'], word_map, max_resp_len, args['task'], args['learning_rate'],
             args['layer_num'], args['drop'])

for epoch in range(args['epochs']):
    print('\nEpoch {}: '.format(epoch))
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar:
        model.train_batch(data, args['grad_threshold'], i == 0)
        pbar.set_description(model.print_loss())

    if (epoch + 1) % args['eval_period'] == 0:
        #2020.1.4 debug到这里
        metrics_score = model.evaluate(dev_loader, metrics_best)
        model.scheduler.step(metrics_score)

        if metrics_score >= metrics_best:
            metrics_best = metrics_score
            cnt = 0
        else:
            cnt += 1
        if cnt > args['patience']:
            print("Ran out of patient, early stop...")
            break
        if math.fabs(metrics_best - 1.0) < 0.001:
            print('Results are so good, early stop...')
            break


