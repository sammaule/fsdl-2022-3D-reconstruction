"""
Source: assessed on 27/09/2022 from:
https://github.com/sunset1995/HorizonNet/blob/master/misc/utils.py
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


def group_weight(module):
    # Group module parameters into two group
    # One need weight_decay and the other doesn't
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [dict(params=group_decay), dict(params=group_no_decay, weight_decay=0.0)]


def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (
            args.max_iters - args.warmup_iters
        )
        scale_running_lr = max((1.0 - frac), 0.0) ** args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = args.running_lr


def save_model(net, path, args):
    state_dict = OrderedDict(
        {
            "args": args.__dict__,
            "kwargs": {
                "backbone": net.backbone,
                "use_rnn": net.use_rnn,
            },
            "state_dict": net.state_dict(),
        }
    )
    torch.save(state_dict, path)


def load_trained_model(Net, path):
    state_dict = torch.load(path, map_location="cpu")
    net = Net(**state_dict["kwargs"])
    state_dict["state_dict"]["x_mean"] = torch.FloatTensor(
        np.array([0.485, 0.456, 0.406])[None, :, None, None]
    )
    state_dict["state_dict"]["x_std"] = torch.FloatTensor(
        np.array([0.229, 0.224, 0.225])[None, :, None, None]
    )

    net.load_state_dict(state_dict["state_dict"], strict=True)
    return net
