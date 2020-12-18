import argparse

parser = argparse.ArgumentParser(description = "parameters")
parser.add_argument('dataset')
args = vars(parser.parse_args())


MEM_TOKEN_SIZE = 6 if args["dataset"] == 'kvr' else 4