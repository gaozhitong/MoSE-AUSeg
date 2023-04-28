import argparse

def parse_args(description=''):
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, default = '', help="Path to experiment config file")
    parser.add_argument("--demo", type=str, default='', help = 'evaluate the data in [train, val, test]')
    args = parser.parse_args()

    return args

args = parse_args()
