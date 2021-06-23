import warnings

import torch.nn as nn
from mmcls.models import ClsHead
from mmcls.models.builder import HEADS, build_loss
from mmcls.models.losses import Accuracy
from mmcv.cnn import constant_init, normal_init

from .fc_module import FcModule


@HEADS.register_module()
class LinearReIDHead(ClsHead):
    """Linear head for re-identification.

    Args:
        num_fcs (int): Number of fcs.
        in_channels (int): Number of channels in the input.
        fc_channels (int): Number of channels in the fcs.
        out_channels (int): Number of channels in the output.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to None.
        num_classes (int, optional): Number of the identities. Default to None.
        loss_cls (dict, optional): Cross entropy loss to train the
            re-identificaiton module.
        loss_triplet (dict, optional): Triplet loss to train the
            re-identificaiton module.
        topk (int, optional): Calculate topk accuracy.
    """

    def __init__(self,
                 num_fcs,
                 in_channels,
                 fc_channels,
                 out_channels,
                 norm_cfg=None,
                 act_cfg=None,
                 num_classes=None,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_triplet=dict(
                     type='TripletLoss', margin=0.3, loss_weight=1.0),
                 cal_acc=False,
                 topk=(1, )):
        super(LinearReIDHead, self).__init__(
            loss=dict(type='CrossEntropyLoss'), topk=topk)
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.compute_accuracy = Accuracy(topk=self.topk)
        if not loss_cls:
            if isinstance(num_classes, int):
                warnings.warn('Since cross entropy is not set, '
                              'the num_classes will be ignored.')
            if cal_acc:
                warnings.warn('Since cross entropy is not set, '
                              'the cal_acc will be ignored.')
            if not loss_triplet:
                raise ValueError('Please choose at least one loss in '
                                 'triplet loss and cross entropy loss.')
        elif not isinstance(num_classes, int):
            raise ValueError('The num_classes must be a current number, '
                             'if there is cross entropy loss.')
        self.compute_loss_cls = build_loss(loss_cls) if loss_cls else None
        self.compute_loss_triplet = build_loss(
            loss_triplet) if loss_triplet else None
        self.cal_acc = cal_acc

        self._init_layers()

    def _init_layers(self):
        """Initialize fc layers."""
        self.fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            in_channels = self.in_channels if i == 0 else self.fc_channels
            self.fcs.append(
                FcModule(in_channels, self.fc_channels, self.norm_cfg,
                         self.act_cfg))
        in_channels = self.in_channels if self.num_fcs == 0 else \
            self.fc_channels
        self.fc_out = nn.Linear(in_channels, self.out_channels)
        if self.num_classes:
            self.bn = nn.BatchNorm1d(self.out_channels)
            self.classifier = nn.Linear(self.out_channels, self.num_classes)

    def init_weights(self):
        """Initalize model weights."""
        normal_init(self.fc_out, mean=0, std=0.01, bias=0)
        if self.num_classes:
            constant_init(self.bn, 1, bias=0)
            normal_init(self.classifier, mean=0, std=0.01, bias=0)

    def simple_test(self, x):
        """Test without augmentation."""
        for m in self.fcs:
            x = m(x)
        fea = self.fc_out(x)
        return fea

    def forward_train(self, x, gt_label):
        """Model forward."""
        for m in self.fcs:
            x = m(x)
        fea = self.fc_out(x)
        losses = dict()
        if not self.compute_loss_cls:
            losses['loss'] = self.compute_loss_triplet(fea, gt_label)
        else:
            fea_bn = self.bn(fea)
            out = self.classifier(fea_bn)
            if self.compute_loss_triplet:
                losses['triplet_loss'] = self.compute_loss_triplet(
                    fea, gt_label)
                losses['ce_loss'] = self.compute_loss_cls(out, gt_label)
            else:
                losses['loss'] = self.compute_loss_cls(out, gt_label)
            if self.cal_acc:
                # compute accuracy
                acc = self.compute_accuracy(out, gt_label)
                assert len(acc) == len(self.topk)
                losses['accuracy'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        return losses
