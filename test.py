import argparse

parser = argparse.ArgumentParser()
a = parser.add_argument("--do_train", default='yes', type=str, required=True,
                        help="Whether to run training. yes or no.")
args = parser.parse_args()

if args.do_train:
    print('do_train is Ture.')
else:
    print('do_train is False.')
