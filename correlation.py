import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
from src.utils.utilities import *
from src.metrics.swap import SWAP
from src.datasets.utilities import get_datasets
from src.search_space.networks import *

# Settings for console outputs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()

# general setting
parser.add_argument('--data_path', default="datasets", type=str, nargs='?', help='path to the image dataset (datasets or datasets/ILSVRC/Data/CLS-LOC)')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--device', default="mps", type=str, nargs='?', help='setup device (cpu, mps or cuda)')
parser.add_argument('--repeats', default=32, type=int, nargs='?', help='times of calculating the training-free metric')
parser.add_argument('--input_samples', default=16, type=int, nargs='?', help='input batch size for training-free metric')

args = parser.parse_args()

if __name__ == "__main__":
    
    device = torch.device(args.device)

    arch_info = pd.read_csv(args.data_path+'/DARTS_archs_CIFAR10.csv', names=['genotype', 'valid_acc'], sep=',')
    
    train_data, _, _ = get_datasets('cifar10', args.data_path, (args.input_samples, 3, 32, 32), -1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.input_samples, num_workers=0, pin_memory=True)
    loader = iter(train_loader)
    inputs, _ = next(loader)  

    results = []
    
    for index, i in arch_info.iterrows():
        print(f'Evaluating network: {index}')

        network = Network(3, 10, 1, eval(i.genotype))
        network = network.to(device)

        swap = SWAP(model=network, inputs=inputs, device=device, seed=args.seed)

        swap_score = []

        for _ in range(args.repeats):
            network = network.apply(network_weight_gaussian_init)
            swap.reinit()
            swap_score.append(swap.forward())
            swap.clear()

        results.append([np.mean(swap_score), i.valid_acc])

    results = pd.DataFrame(results, columns=['swap_score', 'valid_acc'])
    print()    
    print(f'Spearman\'s Correlation Coefficient: {stats.spearmanr(results.swap_score, results.valid_acc)[0]}')
    


