from .operations import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    x = nn.functional.dropout(x, p=drop_prob)

  return x

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, 1, True)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, 1, True)
    
    if reduction:
        op_names, indices = zip(*genotype.reduce)
        concat = genotype.reduce_concat # 2,3,4,5
    else:
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat # 2,3,4,5
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2 # 4
    self._concat = concat # 2,3,4,5
    self.multiplier = len(concat) # 4
    self._ops = nn.ModuleList()

    for name, index in zip(op_names, indices):
        stride = 2 if reduction and index < 2 else 1
        op = OPS[name](C, C, stride, True)
        self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class Network(nn.Module):

    def __init__(self, C, num_classes, layers, genotype):
        self.drop_path_prob = 0.
        super(Network, self).__init__()
        
        self._layers = layers

        C_prev_prev, C_prev, C_curr = C, C, C
        
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = input
        
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out

