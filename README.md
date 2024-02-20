# Sample-Wise Activation Patterns for Ultra-Fast NAS <br/> (ICLR 2024 Spotlight)
Training-free metrics (a.k.a. zero-cost proxies) are widely used to avoid resource-intensive neural network training, especially in Neural Architecture Search (NAS). Recent studies show that existing training-free metrics have several limitations, such as limited correlation and poor generalisation across different search spaces and tasks. Hence, we propose Sample-Wise Activation Patterns and its derivative, SWAP-Score, a novel high-performance training-free metric. It measures the expressivity of networks over a batch of input samples. The SWAP-Score is strongly correlated with ground-truth performance across various search spaces and tasks, outperforming 15 existing training-free metrics on NAS-Bench-101/201/301 and TransNAS-Bench-101.

# Usage

The following instruction demonstrates the usage of evaluating network's performance through SWAP-Score.

**/src/metrics/swap.py** contains the core components of SWAP-Score. 

**/datasets/DARTS_archs_CIFAR10.csv** contains 1000 architectures (randomly sampled from DARTS space) along with their CIFAR-10 validation accuracies (trained for 200 epochs).

* Install necessary dependencies (a new virtual environment is suggested).
```
cd SWAP
pip install -r requirements.txt
```
* Calculate the correlation between SWAP-Score and CIFAR-10 validation accuracies of 1000 DARTS architectures.
```
python correlation.py
```


If you use or build on our code, please consider citing our paper:
```
@inproceedings{
peng2024swapnas,
title={{SWAP}-{NAS}: Sample-Wise Activation Patterns for Ultra-fast {NAS}},
author={Yameng Peng and Andy Song and Haytham M. Fayek and Vic Ciesielski and Xiaojun Chang},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tveiUXU2aa}
}
```
