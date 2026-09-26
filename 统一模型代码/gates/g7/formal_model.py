"""Clean G7 models: random constructors, explicit same-protocol handoffs only.

Architecture/loss code is reused, but no historical model-loading factory is
called. The caller owns file identity checks for supplied new formal states.
"""
from __future__ import annotations

import itertools

import torch
from torch import nn

from 统一模型代码.gates.g7.compact_foundation import FoundationModel as NativeFoundation
from 统一模型代码.gates.g7.compact_foundation import NATIVE_CONFIG
from 统一模型代码.gates.g7.compact_model import CompactModel, g4
from 统一模型代码.gates.g6.g6_p2_model import SplitPhysical
from 统一模型代码.gates.g7.g7_model import LocalFrequencySelector, LocalRefiner
from 统一模型代码.models.e2e_latent_fusion import (
    SourceQueryBuilder, FrequencySpatialSplitter, SourceLocalizationHead,
)
from train_v26 import SourceDetectionNet
from yolo_model import YOLOv8Loc


PROTOCOL = 'G7_FORMAL_FROM_SCRATCH_V1'
CH3_ARCHITECTURE = dict(n_sub=19, max_src=10, mode='transformer')
FORMAL_NATIVE_CONFIG = dict(NATIVE_CONFIG, learning_rate=1e-3,
    initialization='random_from_scratch', ch3_architecture=CH3_ARCHITECTURE)


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    return value


def _new_ch3(device):
    return SourceDetectionNet(**CH3_ARCHITECTURE).to(device)


def _new_d8(device):
    return YOLOv8Loc(method='dualhead', dropout=.4, grad_alpha=1.).to(device)


def _check_foundation(state, kind):
    if (state.get('formal_protocol') != PROTOCOL or state.get('kind') != kind
            or state.get('native_config') != FORMAL_NATIVE_CONFIG):
        raise ValueError('Only a new formal native foundation state is accepted')


class FormalFoundation(NativeFoundation):
    """Native-task scratch training; all native heads, modules and BN buffers."""
    def __init__(self, out, manifest, seed, kind, device='cuda'):
        if kind not in ('ch3', 'd8'):
            raise ValueError(kind)
        g4.set_deterministic(seed)
        self.kind, self.device, self.seed = kind, torch.device(device), int(seed)
        self.model = _new_ch3(self.device) if kind == 'ch3' else _new_d8(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.mode(True)

    def state(self):
        return dict(formal_protocol=PROTOCOL, seed=self.seed, kind=self.kind,
                    model=_clone_tree(self.model.state_dict()),
                    native_config=_clone_tree(FORMAL_NATIVE_CONFIG))

    def restore(self, state):
        _check_foundation(state, self.kind)
        if int(state['seed']) != self.seed:
            raise ValueError('Foundation seed differs')
        self.model.load_state_dict(state['model'], strict=True)

    def optimizer(self, manifest=None, lr=None):
        rate = FORMAL_NATIVE_CONFIG['learning_rate'] if lr is None else float(lr)
        return torch.optim.AdamW(self.model.parameters(), lr=rate,
            weight_decay=FORMAL_NATIVE_CONFIG[f'{self.kind}_weight_decay'])


class FormalCandidate(CompactModel):
    """Fresh source modules on explicitly supplied new CH3/D8 foundations.

    Constructor makes no file reads. ``initial`` may be a new formal candidate
    state, or {'ch3': foundation_state, 'd8': foundation_state}. Without either,
    the model is random, suitable only for engineering checks until foundations
    are handed off. All arms load the same candidate state before local training.
    """
    def __init__(self, out, manifest, seed, initial=None, device='cuda'):
        g4.set_deterministic(seed)
        self.device, self.seed = torch.device(device), int(seed)
        ch3, d8 = _new_ch3(self.device), _new_d8(self.device)
        for p in itertools.chain(ch3.parameters(), d8.parameters()):
            p.requires_grad_(False)
        query = SourceQueryBuilder().to(self.device)
        splitter = FrequencySpatialSplitter().to(self.device)
        head = SourceLocalizationHead().to(self.device)
        nn.init.constant_(head.heatmap.bias, -2.19)
        nn.init.zeros_(head.offset.weight)
        nn.init.zeros_(head.offset.bias)
        groups = [
            dict(params=list(query.parameters()) + list(splitter.parameters()) + list(head.parameters()),
                 lr=1e-3, name='new_candidate'),
            dict(params=list(ch3.band_heads[:3].parameters()), lr=1e-4, name='band_heads'),
            dict(params=list(d8.decoder.c1.parameters()) + list(d8.decoder.up0.parameters()),
                 lr=2e-5, name='d8_tail'),
            dict(params=list(ch3.cross_attn.parameters()), lr=1e-5, name='ch3_tail'),
        ]
        parameters = [p for group in groups for p in group['params']]
        for p in parameters:
            p.requires_grad_(True)
        self.context = g4.Context(ch3, d8, query, splitter, head, 'a2_joint_tail', groups, parameters)
        self.physical = SplitPhysical('b').to(self.device)
        self.selector = LocalFrequencySelector().to(self.device)
        self.refiner = LocalRefiner().to(self.device)
        self.foundations_loaded = False
        self.local_initialized = False
        if initial is not None:
            if set(initial) == {'ch3', 'd8'}:
                self.load_foundations(initial['ch3'], initial['d8'])
            else:
                self.restore(initial.get('state', initial))
        self.mode(True)

    def load_foundations(self, ch3_state, d8_state):
        for kind, state in [('ch3', ch3_state), ('d8', d8_state)]:
            _check_foundation(state, kind)
            if int(state['seed']) != self.seed:
                raise ValueError('Foundation/candidate seed differs')
        self.context.ch3.load_state_dict(ch3_state['model'], strict=True)
        self.context.d8.load_state_dict(d8_state['model'], strict=True)
        self.foundations_loaded = True

    def initialize_local(self):
        """Call once AFTER best-candidate restore, BEFORE saving shared F/S/E init.

        Transfer the learned candidate physical selector. Reset the local refiner
        with a seed independent of epoch/RNG consumption; all F/S/E starts match.
        """
        self.selector.selector.load_state_dict(self.physical.selector.state_dict(), strict=True)
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == 'cuda' else []):
            torch.manual_seed(self.seed + 2000)
            self.refiner = LocalRefiner().to(self.device)
        self.refiner.train(self.context.source_head.training)
        self.local_initialized = True

    def state(self):
        return _clone_tree(dict(formal_protocol=PROTOCOL, seed=self.seed,
            foundations_loaded=self.foundations_loaded, local_initialized=self.local_initialized,
            full_ch3=self.context.ch3.state_dict(), full_d8=self.context.d8.state_dict(),
            base=g4.state_payload(self.context), physical=self.physical.state_dict(),
            selector=self.selector.state_dict(), refiner=self.refiner.state_dict()))

    def restore(self, state):
        if state.get('formal_protocol') != PROTOCOL or int(state.get('seed', -1)) != self.seed:
            raise ValueError('Only same-seed new formal candidate state is accepted')
        # Full native state LAST prevents partial-head payload overriding foundations.
        g4.load_state(self.context, state['base'])
        self.context.ch3.load_state_dict(state['full_ch3'], strict=True)
        self.context.d8.load_state_dict(state['full_d8'], strict=True)
        for name in ('physical', 'selector', 'refiner'):
            getattr(self, name).load_state_dict(state[name], strict=True)
        self.foundations_loaded = bool(state['foundations_loaded'])
        self.local_initialized = bool(state['local_initialized'])

    def optimizer(self, phase, manifest=None):
        if phase in ('baseline', 'candidate'):
            groups = [dict(group) for group in self.context.parameter_groups]
            groups.append(dict(params=list(self.physical.parameters()), lr=1e-3, name='physical'))
        elif phase in ('local', 'f', 's', 'e', 'F', 'S', 'E'):
            groups = [
                dict(params=list(self.context.ch3.band_heads[:3].parameters()) +
                     list(self.context.query_builder.parameters()), lr=1e-4, name='query'),
                dict(params=list(self.context.ch3.cross_attn.parameters()), lr=1e-5, name='ch3_tail'),
                dict(params=list(self.selector.parameters()) + list(self.refiner.parameters()),
                     lr=1e-3, name='local'),
            ]
        else:
            raise ValueError(f'Unknown formal phase: {phase}')
        decay = .005 if manifest is None else float(manifest['config']['weight_decay'])
        return torch.optim.AdamW(groups, weight_decay=decay)


# Explicit aliases ease integration without calling warm-start constructors.
FoundationModel = FormalFoundation
CandidateModel = FormalCandidate
