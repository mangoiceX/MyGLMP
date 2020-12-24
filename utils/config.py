import argparse

UNK_token = 0
PAD_token = 1
SOS_token = 3
EOS_token = 2

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser(description = "parameters")
parser.add_argument('dataset')
args = vars(parser.parse_args())


MEM_TOKEN_SIZE = 6 if args["dataset"] == 'kvr' else 4