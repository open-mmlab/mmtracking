# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import BaseModule

from ..builder import MOTION


@MOTION.register_module()
class FlowNetSimple(BaseModule):
    """The simple version of FlowNet.

    This FlowNetSimple is the implementation of `FlowNetSimple
    <https://arxiv.org/abs/1504.06852>`_.

    Args:
        img_scale_factor (float): Used to upsample/downsample the image.
        out_indices (list): The indices of outputting feature maps after
            each group of conv layers. Defaults to [2, 3, 4, 5, 6].
        flow_scale_factor (float): Used to enlarge the values of flow.
            Defaults to 5.0.
        flow_img_norm_std (list): Used to scale the values of image.
            Defaults to [255.0, 255.0, 255.0].
        flow_img_norm_mean (list): Used to center the values of image.
            Defaults to [0.411, 0.432, 0.450].
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    arch_setting = {
        'conv_layers': {
            'inplanes': (6, 64, 128, 256, 512, 512),
            'kernel_size': (7, 5, 5, 3, 3, 3),
            'num_convs': (1, 1, 2, 2, 2, 2)
        },
        'deconv_layers': {
            'inplanes': (386, 770, 1026, 1024)
        }
    }

    def __init__(self,
                 img_scale_factor,
                 out_indices=[2, 3, 4, 5, 6],
                 flow_scale_factor=5.0,
                 flow_img_norm_std=[255.0, 255.0, 255.0],
                 flow_img_norm_mean=[0.411, 0.432, 0.450],
                 init_cfg=None):
        super(FlowNetSimple, self).__init__(init_cfg)
        self.img_scale_factor = img_scale_factor
        self.out_indices = out_indices
        self.flow_scale_factor = flow_scale_factor
        self.flow_img_norm_mean = flow_img_norm_mean
        self.flow_img_norm_std = flow_img_norm_std

        self.conv_layers = []
        conv_layers_setting = self.arch_setting['conv_layers']
        for i in range(len(conv_layers_setting['inplanes'])):
            num_convs = conv_layers_setting['num_convs'][i]
            kernel_size = conv_layers_setting['kernel_size'][i]
            inplanes = conv_layers_setting['inplanes'][i]
            if i == len(conv_layers_setting['inplanes']) - 1:
                planes = 2 * inplanes
            else:
                planes = conv_layers_setting['inplanes'][i + 1]

            conv_layer = nn.ModuleList()
            conv_layer.append(
                ConvModule(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=(kernel_size - 1) // 2,
                    bias=True,
                    conv_cfg=dict(type='Conv'),
                    act_cfg=dict(type='LeakyReLU', negative_slope=0.1)))
            for j in range(1, num_convs):
                kernel_size = 3 if i == 2 else kernel_size
                conv_layer.append(
                    ConvModule(
                        in_channels=planes,
                        out_channels=planes,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=True,
                        conv_cfg=dict(type='Conv'),
                        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)))

            self.add_module(f'conv{i+1}', conv_layer)
            self.conv_layers.append(f'conv{i+1}')

        self.deconv_layers = []
        self.flow_layers = []
        self.upflow_layers = []
        deconv_layers_setting = self.arch_setting['deconv_layers']
        planes = deconv_layers_setting['inplanes'][-1] // 2
        for i in range(len(deconv_layers_setting['inplanes']) - 1, -1, -1):
            inplanes = deconv_layers_setting['inplanes'][i]

            deconv_layer = ConvModule(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=dict(type='deconv'),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
            self.add_module(f'deconv{i+2}', deconv_layer)
            self.deconv_layers.insert(0, f'deconv{i+2}')

            flow_layer = ConvModule(
                in_channels=inplanes,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=dict(type='Conv'),
                act_cfg=None)
            self.add_module(f'predict_flow{i+3}', flow_layer)
            self.flow_layers.insert(0, f'predict_flow{i+3}')

            upflow_layer = ConvModule(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=dict(type='deconv'),
                act_cfg=None)
            self.add_module(f'upsample_flow{i+2}', upflow_layer)
            self.upflow_layers.insert(0, f'upsample_flow{i+2}')
            planes = planes // 2

        self.predict_flow = ConvModule(
            in_channels=planes * (2 + 4) + 2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv'),
            act_cfg=None)

    def prepare_imgs(self, imgs, img_metas):
        """Preprocess images pairs for computing flow.

        Args:
            imgs (Tensor): of shape (N, 6, H, W) encoding input images pairs.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

        Returns:
            Tensor: of shape (N, 6, H, W) encoding the input images pairs for
            FlowNetSimple.
        """
        if not hasattr(self, 'img_norm_mean'):
            mean = img_metas[0]['img_norm_cfg']['mean']
            mean = torch.tensor(mean, dtype=imgs.dtype, device=imgs.device)
            self.img_norm_mean = mean.repeat(2)[None, :, None, None]

            mean = self.flow_img_norm_mean
            mean = torch.tensor(mean, dtype=imgs.dtype, device=imgs.device)
            self.flow_img_norm_mean = mean.repeat(2)[None, :, None, None]

        if not hasattr(self, 'img_norm_std'):
            std = img_metas[0]['img_norm_cfg']['std']
            std = torch.tensor(std, dtype=imgs.dtype, device=imgs.device)
            self.img_norm_std = std.repeat(2)[None, :, None, None]

            std = self.flow_img_norm_std
            std = torch.tensor(std, dtype=imgs.dtype, device=imgs.device)
            self.flow_img_norm_std = std.repeat(2)[None, :, None, None]

        flow_img = imgs * self.img_norm_std + self.img_norm_mean
        flow_img = flow_img / self.flow_img_norm_std - self.flow_img_norm_mean
        flow_img[:, :, img_metas[0]['img_shape'][0]:, :] = 0.0
        flow_img[:, :, :, img_metas[0]['img_shape'][1]:] = 0.0
        flow_img = torch.nn.functional.interpolate(
            flow_img,
            scale_factor=self.img_scale_factor,
            mode='bilinear',
            align_corners=False)
        return flow_img

    def forward(self, imgs, img_metas):
        """Compute the flow of images pairs.

        Args:
            imgs (Tensor): of shape (N, 6, H, W) encoding input images pairs.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

        Returns:
            Tensor: of shape (N, 2, H, W) encoding flow of images pairs.
        """
        x = self.prepare_imgs(imgs, img_metas)
        conv_outs = []
        for i, conv_name in enumerate(self.conv_layers, 1):
            conv_layer = getattr(self, conv_name)
            for module in conv_layer:
                x = module(x)
            if i in self.out_indices:
                conv_outs.append(x)

        num_outs = len(conv_outs)
        for i, deconv_name, flow_name, upflow_name in zip(
                range(1, num_outs)[::-1], self.deconv_layers[::-1],
                self.flow_layers[::-1], self.upflow_layers[::-1]):
            deconv_layer = getattr(self, deconv_name)
            flow_layer = getattr(self, flow_name)
            upflow_layer = getattr(self, upflow_name)

            if i == num_outs - 1:
                concat_out = conv_outs[i]
            flow = flow_layer(concat_out)
            upflow = self.crop_like(upflow_layer(flow), conv_outs[i - 1])
            deconv_out = self.crop_like(
                deconv_layer(concat_out), conv_outs[i - 1])
            concat_out = torch.cat((conv_outs[i - 1], deconv_out, upflow),
                                   dim=1)

        flow = self.predict_flow(concat_out)
        flow = torch.nn.functional.interpolate(
            flow,
            scale_factor=4 / self.img_scale_factor,
            mode='bilinear',
            align_corners=False)
        flow *= 4 / self.img_scale_factor
        flow *= self.flow_scale_factor

        return flow

    def crop_like(self, input, target):
        """Crop `input` as the size of `target`."""
        if input.size()[2:] == target.size()[2:]:
            return input
        else:
            return input[:, :, :target.size(2), :target.size(3)]
