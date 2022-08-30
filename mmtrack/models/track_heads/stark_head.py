# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.layers import Transformer
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmtrack.registry import MODELS
from mmtrack.utils import InstanceList, OptConfigType, SampleList


@MODELS.register_module()
class CornerPredictorHead(BaseModule):
    """Corner Predictor head.

    Args:
        inplanes (int): input channel
        channel (int): the output channel of the first conv block
        feat_size (int): the size of feature map
        stride (int): the stride of feature map from the backbone
    """

    def __init__(self,
                 inplanes: int,
                 channel: int,
                 feat_size: int = 20,
                 stride: int = 16):
        super(CornerPredictorHead, self).__init__()
        self.feat_size = feat_size
        self.stride = stride
        self.img_size = self.feat_size * self.stride

        def conv_module(in_planes: int,
                        out_planes: int,
                        kernel_size: int = 3,
                        padding: int = 1):
            # The module's pipeline: Conv -> BN -> ReLU.
            return ConvModule(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU'),
                inplace=True)

        # top-left corner
        self.tl_corner_pred = nn.Sequential(
            conv_module(inplanes, channel), conv_module(channel, channel // 2),
            conv_module(channel // 2, channel // 4),
            conv_module(channel // 4, channel // 8),
            nn.Conv2d(channel // 8, 1, kernel_size=1))
        # bottom-right corner
        self.br_corner_pred = nn.Sequential(
            conv_module(inplanes, channel), conv_module(channel, channel // 2),
            conv_module(channel // 2, channel // 4),
            conv_module(channel // 4, channel // 8),
            nn.Conv2d(channel // 8, 1, kernel_size=1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with input x.

        Args:
            x (Tensor): of shape (bs, C, H, W).
        Returns:
            (Tensor): bbox of shape (bs, 4) in (tl_x, tl_y, br_x, br_y) format.
        """
        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)

    def get_score_map(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Score map branch.

        Args:
            x (Tensor): of shape (bs, C, H, W).
        Returns:
            score_map_tl (Tensor): of shape (bs, 1, H, W). The score map of top
                left corner of tracking bbox.
            score_map_br (Tensor): of shape (bs, 1, H, W). The score map of
                bottom right corner of tracking bbox.
        """
        score_map_tl = self.tl_corner_pred(x)
        score_map_br = self.br_corner_pred(x)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map: Tensor) -> Tuple[Tensor, Tensor]:
        """Get soft-argmax coordinate for the given score map.

        Args:
            score_map (Tensor): the last score map
                in bbox_head branch

        Returns:
            exp_x (Tensor): of shape (bs, 1). The values are in range
                [0, self.feat_size * self.stride]
            exp_y (Tensor): of shape (bs, 1). The values are in range
                [0, self.feat_size * self.stride]
        """
        # (bs, feat_size * feat_size)
        score_vec = score_map.view((-1, self.feat_size * self.feat_size))
        prob_vec = nn.functional.softmax(score_vec, dim=1)

        if not hasattr(self, 'coord_x'):
            # generate coordinates and indexes
            self.indice = torch.arange(
                0, self.feat_size, device=score_map.device).view(
                    -1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_size, 1)) \
                .view((self.feat_size * self.feat_size,)).float()
            self.coord_y = self.indice.repeat((1, self.feat_size)) \
                .view((self.feat_size * self.feat_size,)).float()

        soft_argmax_x = torch.sum((self.coord_x * prob_vec), dim=1)
        soft_argmax_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return soft_argmax_x, soft_argmax_y


@MODELS.register_module()
class ScoreHead(nn.Module):
    """Predict the confidence score of target in current frame.

    Cascade multiple FC layer and empose relu on the output of last layer.

    Args:
        input_dim (int): the dim of input.
        hidden_dim (int): the dim of hidden layers.
        output_dim (int): the dim of output.
        num_layers (int): the number of FC layers.
        use_bn (bool, optional): whether to use BN after each FC layer.
            Defaults to False.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 use_bn: bool = False):
        super(ScoreHead, self).__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        if use_bn:
            self.layers = nn.ModuleList(
                nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                for n, k in zip([input_dim] + hidden_dims, hidden_dims +
                                [output_dim]))
        else:
            self.layers = nn.ModuleList(
                nn.Linear(n, k)
                for n, k in zip([input_dim] + hidden_dims, hidden_dims +
                                [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for `ScoreHead`.

        Args:
            x (Tensor): of shape (1, bs, num_query, c).

        Returns:
            Tensor: of shape (bs * num_query, 1).
        """
        # TODO: Perform sigmoid to the last output here rather than in loss
        # calculation.
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.view(-1, 1)


@MODELS.register_module()
class StarkTransformer(Transformer):
    """The transformer head used in STARK. `STARK.

    <https://arxiv.org/abs/2103.17154>`_.

    This module follows the official DETR implementation.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmengine.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmengine.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmengine.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        encoder: OptConfigType = None,
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):
        super(StarkTransformer, self).__init__(
            encoder=encoder, decoder=decoder, init_cfg=init_cfg)

    def forward(self, x: Tensor, mask: Tensor, query_embed: Tensor,
                pos_embed: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function for `StarkTransformer`.

        The difference with transofrmer module in `MMCV` is the input shape.
        The sizes of template feature maps and search feature maps are
        different. Thus, we must flatten and concatenate them outside this
        module. The `MMCV` flatten the input features inside tranformer module.

        Args:
            x (Tensor): Input query with shape (feats_flatten_len, bs, c)
                where c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape (bs, feats_flatten_len).
            query_embed (Tensor): The query embedding for decoder, with shape
                (num_query, c).
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with shape (feats_flatten_len, bs, c).

            Here, 'feats_flatten_len' = z_feat_h*z_feat_w*2 + \
                x_feat_h*x_feat_w.
            'z_feat_h' and 'z_feat_w' denote the height and width of the
            template features respectively.
            'x_feat_h' and 'x_feat_w' denote the height and width of search
            features respectively.
        Returns:
            tuple[Tensor, Tensor]: results of decoder containing the following
                tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True, output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                      Here, return_intermediate_dec=False
                - enc_mem: Output results from encoder, with shape \
                      (feats_flatten_len, bs, embed_dims).
        """
        _, bs, _ = x.shape
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, embed_dims] -> [num_query, bs, embed_dims]

        enc_mem = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_dec_layers, num_query, bs, embed_dims]
        out_dec = self.decoder(
            query=target,
            key=enc_mem,
            value=enc_mem,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask)
        out_dec = out_dec.transpose(1, 2)
        return out_dec, enc_mem


@MODELS.register_module()
class StarkHead(BaseModule):
    """STARK head module for bounding box regression and prediction of
    confidence score of tracking bbox.

    This module is proposed in
    "Learning Spatio-Temporal Transformer for Visual Tracking".
    `STARK <https://arxiv.org/abs/2103.17154>`_.

    Args:
        num_query (int): Number of query in transformer.
        transformer (obj:`mmengine.ConfigDict`|dict): Config for transformer.
            Default: None.
        positional_encoding (obj:`mmengine.ConfigDict`|dict):
            Config for position encoding.
        bbox_head (obj:`mmengine.ConfigDict`|dict, optional): Config for bbox
            head. Defaults to None.
        cls_head (obj:`mmengine.ConfigDict`|dict, optional): Config for
            classification head. Defaults to None.
        loss_cls (obj:`mmengine.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmengine.ConfigDict`|dict): Config of the bbox
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmengine.ConfigDict`|dict): Config of the bbox
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmengine.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmengine.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_query=1,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 bbox_head=None,
                 cls_head=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                 ),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 frozen_modules=None,
                 **kwargs):
        super(StarkHead, self).__init__(init_cfg=init_cfg)
        self.transformer = MODELS.build(transformer)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        assert bbox_head is not None
        self.bbox_head = MODELS.build(bbox_head)
        if cls_head is None:
            # the stage-1 training
            self.loss_bbox = MODELS.build(loss_bbox)
            self.loss_iou = MODELS.build(loss_iou)
            self.cls_head = None
        else:
            # the stage-2 training
            self.cls_head = MODELS.build(cls_head)
            self.loss_cls = MODELS.build(loss_cls)
        self.embed_dims = self.transformer.embed_dims
        self.num_query = num_query
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        if frozen_modules is not None:
            assert isinstance(frozen_modules, list)
            for module in frozen_modules:
                m = getattr(self, module)
                # TODO: Study the influence of freezing BN running_mean and
                # running_variance of `frozen_modules` in the 2nd stage train.
                # The official code doesn't freeze these.
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Parameters initialization."""
        self.transformer.init_weights()

    def _merge_template_search(self, inputs: List[Dict[str, Tensor]]) -> dict:
        """Merge the data of template and search images.
        The merge includes 3 steps: flatten, premute and concatenate.
        Note: the data of search image must be in the last place.

        args:
            inputs (list[dict(Tensor)]):
                The list contains the data of template and search images.
                The dict is in the following format:
                - 'feat': (N, C, H, W)
                - 'mask': (N, H, W)
                - 'pos_embed': (N, C, H, W)

        Return:
            dict(Tensor):
                - 'feat': in [data_flatten_len, N, C] format
                - 'mask': in [N, data_flatten_len] format
                - 'pos_embed': in [data_flatten_len, N, C]
                    format

                Here, 'data_flatten_len' = z_h*z_w*2 + x_h*x_w.
                'z_h' and 'z_w' denote the height and width of the
                template images respectively.
                'x_h' and 'x_w' denote the height and width of search image
                respectively.
        """
        seq_dict = defaultdict(list)
        # flatten and permute
        for input_dic in inputs:
            for name, x in input_dic.items():
                if name == 'mask':
                    seq_dict[name].append(x.flatten(1))
                else:
                    seq_dict[name].append(
                        x.flatten(2).permute(2, 0, 1).contiguous())
        # concatenate
        for name, x in seq_dict.items():
            if name == 'mask':
                seq_dict[name] = torch.cat(x, dim=1)
            else:
                seq_dict[name] = torch.cat(x, dim=0)
        return seq_dict

    def forward_bbox_head(self, feat: Tensor, enc_mem: Tensor) -> Tensor:
        """
        Args:
            feat (Tensor): output embeddings of decoder, with shape
                (1, bs, num_query, c).
            enc_mem (Tensor): output embeddings of encoder, with shape
                (feats_flatten_len, bs, C)

                Here, 'feats_flatten_len' = z_feat_h*z_feat_w*2 + \
                    x_feat_h*x_feat_w.
                'z_feat_h' and 'z_feat_w' denote the height and width of the
                template features respectively.
                'x_feat_h' and 'x_feat_w' denote the height and width of search
                features respectively.
        Returns:
            Tensor: of shape (bs * num_query, 4). The bbox is in
                [tl_x, tl_y, br_x, br_y] format.
        """
        z_feat_len = self.bbox_head.feat_size**2
        # the output of encoder for the search image
        x_feat = enc_mem[-z_feat_len:].transpose(
            0, 1)  # (bs, x_feat_h*x_feat_w, c)
        dec_embed = feat.squeeze(0).transpose(1, 2)  # (bs, c, num_query)
        attention = torch.matmul(
            x_feat, dec_embed)  # (bs, x_feat_h*x_feat_w, num_query)
        bbox_feat = (x_feat.unsqueeze(-1) * attention.unsqueeze(-2))

        # (bs, x_feat_h*x_feat_w, c, num_query) --> (bs, num_query, c, x_feat_h*x_feat_w) # noqa
        bbox_feat = bbox_feat.permute((0, 3, 2, 1)).contiguous()
        bs, num_query, dim, _ = bbox_feat.size()
        bbox_feat = bbox_feat.view(-1, dim, self.bbox_head.feat_size,
                                   self.bbox_head.feat_size)
        # run the corner prediction head
        outputs_coord = self.bbox_head(bbox_feat)
        return outputs_coord

    def forward(self, inputs: List[dict]) -> dict:
        """"
        Args:
            inputs (list[dict(tuple(Tensor))]): The list contains the
                multi-level features and masks of template or search images.
                    - 'feat': (tuple(Tensor)), the Tensor is of shape
                        (bs, c, h//stride, w//stride).
                    - 'mask': (Tensor), of shape (bs, h, w).

                Here, `h` and `w` denote the height and width of input
                image respectively. `stride` is the stride of feature map.

        Returns:
            dict:
                - 'pred_bboxes': (Tensor) of shape (bs * num_query, 4), in
                    [tl_x, tl_y, br_x, br_y] format
                - 'pred_logit': (Tensor) of shape (bs * num_query, 1)
        """
        # 1. preprocess inputs for transformer
        all_inputs = []
        for input in inputs:
            feat = input['feat'][0]
            feat_size = feat.shape[-2:]
            mask = F.interpolate(
                input['mask'][None].float(), size=feat_size).to(torch.bool)[0]
            pos_embed = self.positional_encoding(mask)
            all_inputs.append(dict(feat=feat, mask=mask, pos_embed=pos_embed))
        all_inputs = self._merge_template_search(all_inputs)

        # 2. forward transformer head
        # outs_dec is in (1, bs, num_query, c) shape
        # enc_mem is in (feats_flatten_len, bs, c) shape
        outs_dec, enc_mem = self.transformer(all_inputs['feat'],
                                             all_inputs['mask'],
                                             self.query_embedding.weight,
                                             all_inputs['pos_embed'])

        # 3. forward bbox head and classification head
        if not self.training:
            pred_logits = None
            if self.cls_head is not None:
                # forward the classification head
                pred_logits = self.cls_head(outs_dec)
            pred_bboxes = self.forward_bbox_head(outs_dec, enc_mem)
        else:
            if self.cls_head is not None:
                # stage-1 training: forward the classification head
                pred_logits = self.cls_head(outs_dec)
                pred_bboxes = None
            else:
                # stage-2 training: forward the box prediction head
                pred_logits = None
                pred_bboxes = self.forward_bbox_head(outs_dec, enc_mem)

        return pred_logits, pred_bboxes

    def predict(self, inputs: List[dict], data_samples: SampleList,
                prev_bbox: Tensor, scale_factor: Tensor) -> InstanceList:
        """Perform forward propagation of the tracking head and predict
        tracking results on the features of the upstream network.

        Args:
            inputs (list[dict(tuple(Tensor))]): The list contains the
                multi-level features and masks of template or search images.
                    - 'feat': (tuple(Tensor)), the Tensor is of shape
                        (bs, c, h//stride, w//stride).
                    - 'mask': (Tensor), of shape (bs, h, w).

                Here, `h` and `w` denote the height and width of input
                image respectively. `stride` is the stride of feature map.

            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (Tensor): scale factor.

        Returns:
            List[:obj:`InstanceData`]: Object tracking results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (1, )
                - bboxes (Tensor): Has a shape (1, 4),
                  the last dimension 4 arrange as [x1, y1, x2, y2].
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in data_samples
        ]
        outs = self(inputs)
        predictions = self.predict_by_feat(
            *outs,
            prev_bbox=prev_bbox,
            scale_factor=scale_factor,
            batch_img_metas=batch_img_metas)
        return predictions

    def predict_by_feat(self, pred_logits: Tensor, pred_bboxes: Tensor,
                        prev_bbox: Tensor, scale_factor: Tensor,
                        batch_img_metas: List[dict]) -> InstanceList:
        """Track `prev_bbox` to current frame based on the output of network.

        Args:
            pred_logit: (Tensor) of shape (bs * num_query, 1). This item
                only exists when the model has classification head.
            pred_bboxes: (Tensor) of shape (bs * num_query, 4), in
                [tl_x, tl_y, br_x, br_y] format
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (Tensor): scale factor.
            batch_img_metas (List[dict]): the meta information of all images in
                a batch.

        Returns:
            List[:obj:`InstanceData`]: Object tracking results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (1, )
                - bboxes (Tensor): Has a shape (1, 4),
                  the last dimension 4 arrange as [x1, y1, x2, y2].
        """
        result = InstanceData()
        if pred_logits is not None:
            result.scores = pred_logits.view(-1).sigmoid()
        else:
            result.scores = prev_bbox.new_tensor([-1.])
        result.bboxes = pred_bboxes

        return self._bbox_post_process([result],
                                       prev_bbox,
                                       scale_factor,
                                       batch_img_metas=batch_img_metas)

    def _bbox_post_process(self, results: InstanceList, prev_bbox: Tensor,
                           scale_factor: Tensor, batch_img_metas: List[dict],
                           **kwargs) -> InstanceList:
        """The postprocess of tracking bboxes.

        Args:
            results (InstanceList): tracking results.
            prev_bbox (Tensor): of shape (4, ) in [cx, cy, w, h] format.
            scale_factor (Tensor): scale factor.
            batch_img_metas (List[dict]): the meta information of all images in
                a batch.

        Returns:
            List[:obj:`InstanceData`]: Object tracking results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (1, )
                - bboxes (Tensor): Has a shape (1, 4),
                  the last dimension 4 arrange as [x1, y1, x2, y2].
        """
        result = results[0]
        final_bbox = self._mapping_bbox_back(result.bboxes, prev_bbox,
                                             scale_factor)
        img_shape = batch_img_metas[0]['ori_shape']
        final_bbox = self._bbox_clip(
            final_bbox, img_shape[0], img_shape[1], margin=10)
        result.bboxes = final_bbox[None]

        return [result]

    def _mapping_bbox_back(self, pred_bboxes: Tensor, prev_bbox: Tensor,
                           resize_factor: float) -> Tensor:
        """Mapping the `prediction bboxes` from resized cropped image to
        original image. The coordinate origins of them are both the top left
        corner.

        Args:
            pred_bboxes (Tensor): the predicted bbox of shape
                (bs * num_query, 4), in [tl_x, tl_y, br_x, br_y] format.
                The coordinates are based in the resized cropped image.
            prev_bbox (Tensor): the previous bbox of shape (B, 4),
                in [cx, cy, w, h] format. The coordinates are based in the
                original image.
            resize_factor (float): the ratio of original image scale to cropped
                image scale.
        Returns:
            (Tensor): of (4, ) shape, in [tl_x, tl_y, br_x, br_y] format.
        """
        # based in the original croped image
        pred_bbox = pred_bboxes.mean(dim=0) / resize_factor

        # the half size of the original croped image
        cropped_img_half_size = 0.5 * self.test_cfg[
            'search_size'] / resize_factor
        # (x_shift, y_shift) is the coordinate of top left corner of the
        # cropped image based in the original image.
        x_shift, y_shift = prev_bbox[0] - cropped_img_half_size, prev_bbox[
            1] - cropped_img_half_size
        pred_bbox[0:4:2] += x_shift
        pred_bbox[1:4:2] += y_shift

        return pred_bbox

    def _bbox_clip(self,
                   bbox: Tensor,
                   img_h: int,
                   img_w: int,
                   margin: int = 0) -> Tensor:
        """Clip the bbox in [tl_x, tl_y, br_x, br_y] format.

        Args:
            bbox (Tensor): Bounding bbox.
            img_h (int): The height of the image.
            img_w (int): The width of the image.
            margin (int, optional): The distance from image boundary.
                Defaults to 0.

        Returns:
            Tensor: The clipped bounding box.
        """

        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bbox[0] = bbox[0].clamp(0, img_w - margin)
        bbox[1] = bbox[1].clamp(0, img_h - margin)
        bbox_w = bbox_w.clamp(margin, img_w)
        bbox_h = bbox_h.clamp(margin, img_h)
        bbox[2] = bbox[0] + bbox_w
        bbox[3] = bbox[1] + bbox_h
        return bbox

    # TODO: unify the `sefl.predict`, `self.loss` and so on in all the heads of
    # SOT.
    def loss(self, inputs: List[dict], data_samples: SampleList,
             **kwargs) -> dict:
        """Compute loss.

        Args:
            inputs (list[dict(tuple(Tensor))]): The list contains the
                multi-level features and masks of template or search images.
                    - 'feat': (tuple(Tensor)), the Tensor is of shape
                        (bs, c, h//stride, w//stride).
                    - 'mask': (Tensor), of shape (bs, h, w).
                Here, `h` and `w` denote the height and width of input
                image respectively. `stride` is the stride of feature map.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`
                and 'metainfo'.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        outs = self(inputs)

        batch_gt_instances = []
        batch_img_metas = []
        batch_search_gt_instances = []
        for data_sample in data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_search_gt_instances.append(data_sample.search_gt_instances)

        loss_inputs = outs + (batch_gt_instances, batch_search_gt_instances,
                              batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(self, pred_logits: Tensor, pred_bboxes: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_search_gt_instances: InstanceList,
                     batch_img_metas: List[dict]) -> dict:
        """Compute loss.

        Args:
            pred_logits: (Tensor) of shape (bs * num_query, 1). This item
                only exists when the model has classification head.
            pred_bboxes: (Tensor) of shape (bs * num_query, 4), in
                [tl_x, tl_y, br_x, br_y] format
            batch_gt_instances (InstanceList): the instances in a batch.
            batch_search_gt_instances (InstanceList): the search instances in a
                batch.
            batch_img_metas (List[dict]): the meta information of all images in
                a batch.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()
        if self.cls_head is None:
            # the stage-1 training
            assert pred_bboxes is not None
            img_shape = batch_img_metas[0]['search_img_shape']
            pred_bboxes[:, 0:4:2] = pred_bboxes[:, 0:4:2] / float(img_shape[1])
            pred_bboxes[:, 1:4:2] = pred_bboxes[:, 1:4:2] / float(img_shape[0])

            gt_bboxes = [
                instance['bboxes'] for instance in batch_search_gt_instances
            ]
            gt_bboxes = torch.cat(gt_bboxes, dim=0).type(torch.float32)
            gt_bboxes[:, 0:4:2] = gt_bboxes[:, 0:4:2] / float(img_shape[1])
            gt_bboxes[:, 1:4:2] = gt_bboxes[:, 1:4:2] / float(img_shape[0])
            gt_bboxes = gt_bboxes.clamp(0., 1.)

            # regression IoU loss, default GIoU loss
            if (pred_bboxes[:, :2] >= pred_bboxes[:, 2:]).any() or (
                    gt_bboxes[:, :2] >= gt_bboxes[:, 2:]).any():
                # the first several iterations of train may return invalid
                # bbox coordinates.
                losses['loss_iou'] = (pred_bboxes - gt_bboxes).sum() * 0.0
            else:
                losses['loss_iou'] = self.loss_iou(pred_bboxes, gt_bboxes)
            # regression L1 loss
            losses['loss_bbox'] = self.loss_bbox(pred_bboxes, gt_bboxes)
        else:
            # the stage-2 training
            assert pred_logits is not None
            pred_logits = pred_logits.squeeze()

            gt_labels = [
                instance['labels'] for instance in batch_search_gt_instances
            ]
            gt_labels = torch.cat(
                gt_labels, dim=0).type(torch.float32).squeeze()
            losses['loss_cls'] = self.loss_cls(pred_logits, gt_labels)

        return losses
