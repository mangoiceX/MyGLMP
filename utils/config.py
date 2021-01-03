import argparse
import os
import torch


# 常量模块

class Const:
    class ConstError(TypeError): pass
    class ConstCaseError(ConstError): pass

    def __setattr__(self,name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.{}".format(name))
        if not name.isupper():
            raise self.ConstCaseError("const name {} is not all uppercase".format(name))
        self.__dict__[name] = value


const = Const()

UNK_token = 0
PAD_token = 1
SOS_token = 3
EOS_token = 2

if torch.cuda.is_available():
#  if (os.cpu_count() > 8):   # 官方代码为什么这样弄不知道
    USE_CUDA = True
else:
    USE_CUDA = False

# -d=babi -t=1 -h=128 -lr=0.01 -ln=5 -d=0.1 -e=1 -gd=1000 -ep=4 -pt=8 -b=8
parser = argparse.ArgumentParser(description = "parameters")
parser.add_argument('-d', '--dataset',help = 'dataset, babi or kvr')
parser.add_argument('-t', '--task', help='Task Number', type = int)
parser.add_argument('-hsz', '--hidden_size', help = 'Hidden size', type = int)
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate', type = float)
parser.add_argument('-ln', '--layer_num', help = 'Layer Number', type = int)
parser.add_argument('-dr', '--drop', help = 'dropout', type = float)
parser.add_argument('-e', '--epochs', help = 'Epoch Number', type = int)
parser.add_argument('-gd', '--grad_threshold', help = 'Gradient Threshold', type = float)
parser.add_argument('-ep', '--eval_period', help = 'Evaluation period', type = int)
parser.add_argument('-pt','--patience', help = 'Patience', type = int)
parser.add_argument('-b','--batch_size', help='Batch_size', type = int)
parser.add_argument('-abh', '--ablationH', type = bool, required = False, default = False)
parser.add_argument('-abg', '--ablationG', type = bool, required = False, default = False)

args = vars(parser.parse_args())


MEM_TOKEN_SIZE = 6 if args["dataset"] == 'kvr' else 4