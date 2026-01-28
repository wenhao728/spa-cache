import logging

import torch
from torch.utils.flop_counter import FlopCounterMode

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MetricMeter:
    """Meter for tracking metrics.

    Args:
        name (str): The name of the metric.
        mode (str, optional): The mode of the metric. Options ['val', 'sum', 'cnt', 'avg']. Defaults to 'avg'.
        fmt (str, optional): The format string for printing. Defaults to ':.3f'.
        alpha (float, optional): The smoothing factor for avg mode (EMA). Defaults to 0.2.
    """
    _supported_modes = ('val', 'sum', 'cnt', 'avg', 'ema')

    def __init__(self, name: str, mode: str = 'avg', fmt: str = ':.3f', alpha: float = 0.2):
        if mode not in self._supported_modes:
            raise ValueError(f'Unsupported mode={mode}, supported modes are {self._supported_modes}')
        self.name = name
        self.mode = mode
        self.fmt = fmt
        self.alpha = alpha  # smoothing factor for avg mode (EMA), e.g. 0.8 ** 10 -> 0.107
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0
        self.ema = 0

    def update(self, val: float, n: int = 1):
        reset = self.cnt == 0

        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

        if reset:
            self.ema = self.sum / self.cnt
        else:
            _alpha = 1 - (1 - self.alpha) ** n  # compatiable with n > 1
            self.ema = _alpha * self.val + (1 - _alpha) * self.ema

    def __str__(self):
        if self.mode == 'val':
            fmtstr = f'{self.name} {{val{self.fmt}}}'
        else:
            fmtstr = f'{self.name} {{val{self.fmt}}} ({{{self.mode}{self.fmt}}}){self.mode[0]}'
        return fmtstr.format(**self.__dict__)
    
    @property
    def summary(self):
        return self.__getattribute__(self.mode)


class BenchmarkLogger:
    def __init__(
        self,
        log_latency: bool = True,
        log_flops: bool = False,
        mode: str = 'sum',
        latency_fmt: str = ':.2f',
        flops_fmt: str = ':.6e',
        latency_scale: float = 1e-3,
        flops_scale: float = 1e-9,
    ):
        if log_latency and log_flops:
            logger.warning('Both log_latency and log_flops are enabled, only log_latency will be used')
            log_flops = False

        if log_latency:
            self.latency_meter = MetricMeter(name="Latency", mode=mode, fmt=latency_fmt)
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

        if log_flops:
            self.flops_meter = MetricMeter(name="FLOPs", mode=mode, fmt=flops_fmt)
            self.flops_counter = FlopCounterMode(mods=None, depth=None, display=False)

        self.log_latency = log_latency
        self.log_flops = log_flops
        self.latency_scale = latency_scale
        self.flops_scale = flops_scale

    def __enter__(self):

        if self.log_latency:
            self.start_event.record()

        if self.log_flops:
            self.flops_counter.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.log_latency:
            self.end_event.record()
            self.end_event.synchronize()
            latency = self.start_event.elapsed_time(self.end_event) * self.latency_scale  # ms to s
            self.latency_meter.update(latency)

        if self.log_flops:
            self.flops_counter.__exit__(exc_type, exc_value, exc_traceback)
            gflops = self.flops_counter.get_total_flops() * self.flops_scale  # GFlops
            self.flops_meter.update(gflops)

    @property
    def latency(self):
        return self.latency_meter.summary

    @property
    def flops(self):
        return self.flops_meter.summary

    @property
    def result(self):
        if self.log_latency:
            return self.latency  
        elif self.log_flops:
            return self.flops
        else:
            return 0.0

    def reset(self):
        if self.log_latency:
            self.latency_meter.reset()
        if self.log_flops:
            self.flops_meter.reset()