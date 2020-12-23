import argparse

UNK_token = 0
PAD_token = 1
SOS_token = 3
EOS_token = 2


parser = argparse.ArgumentParser(description = "parameters")
parser.add_argument('dataset')
args = vars(parser.parse_args())


MEM_TOKEN_SIZE = 6 if args["dataset"] == 'kvr' else 4