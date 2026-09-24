"""Separate semantic band logits from physical frequency selection."""
import torch
from torch import nn

from 统一模型代码.gates.g5.e2e_g5_model import forward as base_forward
from 统一模型代码.gates.g6.g6_p1_model import PhysicalResidual, identity_hits as identity_hits
from 统一模型代码.gates.g6.g6_p1_speed import physical_maps


class SplitPhysical(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ('a','b','c'):
            raise ValueError('Unknown P2 arm')
        self.physical = PhysicalResidual()
        self.selector = nn.Linear(128,19) if arm != 'a' else None
        if self.selector is not None:
            nn.init.zeros_(self.selector.weight)
            nn.init.zeros_(self.selector.bias)

    def weights(self, query, logits, arm):
        if arm == 'a':
            return logits.detach().sigmoid()
        q = query.detach() if arm == 'b' else query
        return (logits.detach()+self.selector(q)).sigmoid()


def forward(context, head, features, ids, arm, physics, stats):
    query,logits,attention,heat,offset = base_forward(
        context,features,ids,torch.device('cuda:0'),stop_gradient=True)
    weights = head.weights(query,logits,arm)
    residual = head.physical(physical_maps(physics,weights,stats))
    return (query,logits,attention,heat+residual,offset),residual


def subband_counts(logits, weights, bands, ignore, mapping):
    """Use one semantic-band assignment for both branches; count source-band pairs."""
    k = len(mapping)
    active = (bands[:k]>.5) & (ignore[:k]<.5)
    result = {name:0 for name in ('shared_positive','exclusive_positive',
        'semantic_shared_missed','semantic_exclusive_missed','physical_shared_missed','physical_exclusive_missed')}
    for q,t in mapping.items():
        other = active.clone(); other[t] = False
        shared = active[t] & other.any(0)
        exclusive = active[t] & ~other.any(0)
        result['shared_positive'] += int(shared.sum())
        result['exclusive_positive'] += int(exclusive.sum())
        for name,present in [('semantic',logits[q]>=0),('physical',weights[q]>=.5)]:
            result[f'{name}_shared_missed'] += int((shared & ~present).sum())
            result[f'{name}_exclusive_missed'] += int((exclusive & ~present).sum())
    return result
