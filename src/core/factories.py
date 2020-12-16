# from functools import wraps
from inspect import isfunction, isgeneratorfunction, getmembers
from collections.abc import Sequence
from abc import ABC, ABCMeta
from itertools import chain
from importlib import import_module

import torch
import torch.nn as nn
import torch.utils.data as data

from .misc import (R, MODELS, OPTIMS, CRITNS, DATA)


class _AttrDesc:
    def __init__(self, key):
        self.key = key
    def __get__(self, instance, owner):
        return tuple(getattr(ele, self.key) for ele in instance)
    def __set__(self, instance, value):
        for ele in instance:
            setattr(ele, self.key, value)


def _func_deco(func_name):
    # FIXME: The signature of the wrapped function will be lost.
    def _wrapper(self, *args, **kwargs):
        return tuple(getattr(ele, func_name)(*args, **kwargs) for ele in self)
    return _wrapper


def _generator_deco(func_name):
    # FIXME: The signature of the wrapped function will be lost.
    def _wrapper(self, *args, **kwargs):
        for ele in self:
            yield from getattr(ele, func_name)(*args, **kwargs)
    return _wrapper


# Duck typing
class Duck(Sequence, ABC):
    __ducktype__ = object
    def __init__(self, *args):
        if any(not isinstance(arg, self.__ducktype__) for arg in args):
            raise TypeError("Please check the input type.")
        self._seq = tuple(args)

    def __getitem__(self, key):
        return self._seq[key]

    def __len__(self):
        return len(self._seq)

    def __repr__(self):
        return repr(self._seq)


class DuckMeta(ABCMeta):
    def __new__(cls, name, bases, attrs):
        if len(bases) > 1:
            raise NotImplementedError("Multiple inheritance is not yet supported.")
        members = dict(getmembers(bases[0]))  # Trade space for time

        for k in attrs['__ava__']:
            if k in members:
                v = members[k]
                if isgeneratorfunction(v):
                    attrs.setdefault(k, _generator_deco(k))
                elif isfunction(v):
                    attrs.setdefault(k, _func_deco(k))
                else:
                    attrs.setdefault(k, _AttrDesc(k))
        attrs['__ducktype__'] = bases[0]
        return super().__new__(cls, name, (Duck,), attrs)


class DuckModel(nn.Module):
    __ava__ = ('state_dict', 'load_state_dict', 'forward', '__call__', 'train', 'eval', 'to', 'training')
    def __init__(self, *models):
        super().__init__()
        # XXX: The state_dict will be a little larger in size,
        # since some extra bytes are stored in every key.
        self._m = nn.ModuleList(models)

    def __len__(self):
        return len(self._m)

    def __getitem__(self, idx):
        return self._m[idx]

    def __contains__(self, m):
        return m in self._m

    def __repr__(self):
        return repr(self._m)

    def forward(self, *args, **kwargs):
        return tuple(m(*args, **kwargs) for m in self._m)


Duck.register(DuckModel)


class DuckOptimizer(torch.optim.Optimizer, metaclass=DuckMeta):
    __ava__ = ('param_groups', 'state_dict', 'load_state_dict', 'zero_grad', 'step')
    # An instance attribute can not be automatically handled by metaclass
    @property
    def param_groups(self):
        return list(chain.from_iterable(ele.param_groups for ele in self))

    # Sepcial dispatching rule
    def load_state_dict(self, state_dicts):
        for optim, state_dict in zip(self, state_dicts):
            optim.load_state_dict(state_dict)


class DuckCriterion(nn.Module, metaclass=DuckMeta):
    __ava__ = ('forward', '__call__', 'train', 'eval', 'to')
    pass


class DuckDataLoader(data.DataLoader, metaclass=DuckMeta):
    __ava__ = ()
    pass


def _import_module(pkg: str, mod: str, rel=False):
    if not rel:
        # Use absolute import
        return import_module('.'.join([pkg, mod]), package=None)
    else:
        return import_module('.'+mod, package=pkg)


def single_model_factory(model_name, C):
    builder_name = '_'.join([model_name, C['model'], C['dataset'], 'model'])
    if builder_name in MODELS:
        return MODELS[builder_name](C)
    builder_name = '_'.join([model_name, C['dataset'], 'model'])
    if builder_name in MODELS:
        return MODELS[builder_name](C)
    builder_name = '_'.join([model_name, 'model'])
    if builder_name in MODELS:
        return MODELS[builder_name](C)
    else:
        raise NotImplementedError("{} is not a supported architecture.".format(model_name))


def single_optim_factory(optim_name, params, C):
    builder_name = '_'.join([optim_name, 'optim'])
    if builder_name not in OPTIMS:
        raise NotImplementedError("{} is not a supported optimizer type.".format(optim_name))
    return OPTIMS[builder_name](params, C)
        

def single_critn_factory(critn_name, C):
    builder_name = '_'.join([critn_name, 'critn'])
    if builder_name not in CRITNS:
        raise NotImplementedError("{} is not a supported criterion type.".format(critn_name))
    return CRITNS[builder_name](C)
        

def single_data_factory(dataset_name, phase, C):
    builder_name = '_'.join([dataset_name, C['dataset'], C['model'], phase, 'dataset'])
    if builder_name in DATA:
        return DATA[builder_name](C)
    builder_name = '_'.join([dataset_name, C['model'], phase, 'dataset'])
    if builder_name in DATA:
        return DATA[builder_name](C)
    builder_name = '_'.join([dataset_name, phase, 'dataset'])
    if builder_name in DATA:
        return DATA[builder_name](C)
    else:
        raise NotImplementedError("{} is not a supported dataset.".format(dataset_name))


def _parse_input_names(name_str):
    return name_str.split('+')


def model_factory(model_names, C):
    name_list = _parse_input_names(model_names)
    if len(name_list) > 1:
        return DuckModel(*(single_model_factory(name, C) for name in name_list))
    else:
        return single_model_factory(model_names, C)


def optim_factory(optim_names, models, C):
    name_list = _parse_input_names(optim_names)
    num_models = len(models) if isinstance(models, DuckModel) else 1
    if len(name_list) != num_models:
        raise ValueError("The number of optimizers does not match the number of models.")
    
    if num_models > 1:
        optims = []
        for name, model in zip(name_list, models):
            param_groups = [{'params': module.parameters(), 'name': module_name} for module_name, module in model.named_children()]
            if next(model.parameters(recurse=False), None) is not None:
                param_groups.append({'params': model.parameters(recurse=False), 'name': '_direct'})
            optims.append(single_optim_factory(name, param_groups, C))
        return DuckOptimizer(*optims)
    else:
        return single_optim_factory(
            optim_names, 
            [{'params': module.parameters(), 'name': module_name} for module_name, module in models.named_children()] + 
            ([{'params': models.parameters(recurse=False), 'name': '_direct'}] 
            if next(models.parameters(recurse=False), None) is not None else []), 
            C
        )


def critn_factory(critn_names, C):
    name_list = _parse_input_names(critn_names)
    if len(name_list) > 1:
        return DuckCriterion(*(single_critn_factory(name, C) for name in name_list))
    else:
        return single_critn_factory(critn_names, C)


def data_factory(dataset_names, phase, C):
    name_list = _parse_input_names(dataset_names)
    if len(name_list) > 1:
        return DuckDataLoader(*(single_data_factory(name, phase, C) for name in name_list))
    else:
        return single_data_factory(dataset_names, phase, C)