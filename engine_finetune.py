import math
import sys
from typing import Iterable, Optional
import jax

import flax.core
import flax.linen as fnn
import objax
import objax.nn as nn
import jax.numpy as jnp
from utils import Identity

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: fnn.Module, criterion: fnn.Module,  #from torch.nn.Module
                    data_loader: Iterable, optimizer: jax.example_libraries.optimizers.Optimizer, #from torch.optim.Optimizer
                    device: jax.devices, epoch: int, loss_scaler, max_norm: float = 0,  #from torch.device
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():   #from torch.cuda.amp.autocast
