# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union
import torch

from mydl.config import CfgNode

from torch.optim.lr_scheduler import StepLR
from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR
from .. custom.protein.base import finetune_params

from pampy import match, _

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = cfg.clone()

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer_type: Type[torch.optim.Optimizer], gradient_clipper: _GradientClipper
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                gradient_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer_type.__name__ + "WithGradientClip",
        (optimizer_type,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.Optimizer:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer instance of some type OptimizerType to become an instance
    of the new dynamically created class OptimizerTypeWithGradientClip
    that inherits OptimizerType and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode
            configuration options
        optimizer: torch.optim.Optimizer
            existing optimizer instance

    Return:
        optimizer: torch.optim.Optimizer
            either the unmodified optimizer instance (if gradient clipping is
            disabled), or the same instance with adjusted __class__ to override
            the `step` method and include gradient clipping
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        type(optimizer), grad_clipper
    )
    optimizer.__class__ = OptimizerWithGradientClip
    return optimizer


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if cfg.VERSION != 1 and isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = None

    if cfg.VERSION == 1:
        name = cfg.SOLVER.SCHEDULER
    else:
        name = cfg.SOLVER.LR_SCHEDULER_NAME

    return StepLR( optimizer, step_size=cfg.SOLVER.STEP_SIZE, gamma=cfg.SOLVER.GAMMA)

    """
    # print('scheduler name: {}'.format(name))
    return match
    (
        name,
        "WarmupMultiStepLR",  WarmupMultiStepLR( \
            optimizer, \
            cfg.SOLVER.STEPS, \
            cfg.SOLVER.GAMMA, \
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR, \
            warmup_iters=cfg.SOLVER.WARMUP_ITERS, \
            warmup_method=cfg.SOLVER.WARMUP_METHOD, \
        ),
        "WarmupCosineLR", WarmupCosineLR( \
            optimizer, \
            cfg.SOLVER.MAX_ITER, \
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR, \
            warmup_iters=cfg.SOLVER.WARMUP_ITERS, \
            warmup_method=cfg.SOLVER.WARMUP_METHOD, \
        ),
        "ReduceLROnPlateau", ReduceLROnPlateau( \
            optimizer, \
            factor=cfg.SOLVER.GAMMA, \
            patience=cfg.SOLVER.PATIENCE \
        ),
        "MultiStepLR", MultiStepLR( \
            optimizer, \
            milestones=cfg.SOLVER.STEPS, \
            gamma=cfg.SOLVER.GAMMA \
        ),
        "StepLR", StepLR( \
            optimizer, \
            step_size=cfg.SOLVER.STEP_SIZE, \
            gamma=cfg.SOLVER.GAMMA \
        ),
        "CosineAnnealingLR", CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.T_MAX,
            eta_min=1e-5
        ),
        # match bug exist
        _, StepLR( optimizer, step_size=cfg.SOLVER.STEP_SIZE, gamma=cfg.SOLVER.GAMMA)
        # _, raise ValueError("Unknown LR scheduler: {}".format(name))
    )
    """

def build_finetune_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.FINETUNE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.FINETUNE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        for name in finetune_params[cfg.MODEL.NAME]:
            if name in key:
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr=0, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer

    

