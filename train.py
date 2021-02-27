import argparse

import torch

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True)

parser.add_argument("--model-name", default="transgan_s")
parser.add_argument("--size", default=32, type=int)
parser.add_argument("--dataset", default="c10")
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--g-batch-size", default=32, type=int)
parser.add_argument("--d-batch-size", default=16, type=int)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
args = parser.parse_args()



