import argparse
import os


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

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser(description = "parameters")
parser.add_argument('-d', '--dataset',help = 'dataset, babi or kvr')
parser.add_argument('-t', '--task', help='Task Number', type = int)
parser.add_argument('-b','--batch_size', help='Batch_size', type = int)
args = vars(parser.parse_args())


MEM_TOKEN_SIZE = 6 if args["dataset"] == 'kvr' else 4