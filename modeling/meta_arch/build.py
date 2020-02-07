# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from mydl.utils.registry import Registry

from mydl.custom.protein.network import NetWrapper
import torch

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE

    # version 1
    if meta_arch == 'ProteinResnet':
        device = torch.device(cfg.MODEL.DEVICE)
        model = NetWrapper(cfg)
        model.to(device)
        return torch.nn.DataParallel(model)
    # latest version
    else:
        return META_ARCH_REGISTRY.get(meta_arch)(cfg)
