# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin
from mmengine.optim.scheduler.param_scheduler import INF, _ParamScheduler
from torch.optim import Optimizer

from mmtrack.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class SiamRPNExpParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by exponentially
    changing small multiplicative factor until the number of epoch reaches a
    pre-defined milestone: ``end``.

    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    .. math::

        X_{t} = X_{t-1} \times (\frac{end}{begin})^{\frac{1}{epochs}}


    Args:
        optimizer (Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        start_factor (float): The number we multiply parameter value in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 0.1.
        end_factor (float): The number we multiply parameter value at the end
            of linear changing process. Defaults to 1.0.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        endpoint (bool): If true, `end_factor`` is included in the ``end``.
            Otherwise, it is not included. Default is True.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 start_factor: float = 0.1,
                 end_factor: float = 1.0,
                 begin: int = 0,
                 end: int = INF,
                 endpoint: bool = True,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        if end >= INF:
            raise ValueError('``end`` must be less than infinity,'
                             'Please set ``end`` parameter of '
                             '``QuadraticWarmupScheduler`` as the '
                             'number of warmup end.')

        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(
                'Starting multiplicative factor should between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                'Ending multiplicative factor should between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.endpoint = endpoint
        self.total_iters = end - begin - 1 if self.endpoint else end - begin
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin: int = 0,
                              end: int = INF,
                              by_epoch: bool = True,
                              epoch_length: Optional[int] = None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config.

        Args:
            begin (int, optional): Step at which to start updating the
                parameters. Defaults to 0.
            end (int, optional): Step at which to stop updating the parameters.
                Defaults to INF.
            by_epoch (bool, optional): Whether the scheduled parameters are
                updated by epochs. Defaults to True.
            epoch_length (Optional[int], optional): The length of each epoch.
                Defaults to None.

        Returns:
            Object: The instantiated object of ``SiamRPNExpParamScheduler``.
        """

        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = begin * epoch_length
        if end != INF:
            end = end * epoch_length
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step == 0:
            return [
                group[self.param_name] * self.start_factor
                for group in self.optimizer.param_groups
            ]

        return [
            group[self.param_name] *
            math.pow(self.end_factor / self.start_factor, 1 /
                     (self.total_iters))
            for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class SiamRPNExpLR(LRSchedulerMixin, SiamRPNExpParamScheduler):
    """Decays the parameter value of each parameter group by exponentially
    changing small multiplicative factor until the number of epoch reaches a
    pre-defined milestone: ``end``.

    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    .. math::

        X_{t} = X_{t-1} \times (\frac{end}{begin})^{\frac{1}{epochs}}


    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply parameter value in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 0.1.
        end_factor (float): The number we multiply parameter value at the end
            of linear changing process. Defaults to 1.0.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        endpoint (bool): If true, `end_factor`` is included in the ``end``.
            Otherwise, it is not included. Default is True.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """
