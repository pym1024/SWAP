# Sample-Wise Activation Patterns for Ultra-Fast NAS <br/> (ICLR 2024 Spotlight)
SWAP-Score, based on sample-wise activation patterns, is a metric that accesses the performance of neural networks without training.

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
