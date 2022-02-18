import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner.base_module import BaseModule
from mmdet.models import HEADS
from mmdet.models.builder import build_head, build_loss
from mmdet.models.utils import Transformer, build_transformer
from mmdet.models.utils.builder import TRANSFORMER
from torch import nn


@HEADS.register_module()
class CornerPredictorHead(BaseModule):
    """Corner Predictor head.

    Args:
        inplanes (int): input channel
        channel (int): the output channel of the first conv block
        feat_size (int): the size of feature map
        stride (int): the stride of feature map from the backbone
    """

    def __init__(self, inplanes, channel, feat_size=20, stride=16):
        super(CornerPredictorHead, self).__init__()
        self.feat_size = feat_size
        self.stride = stride
        self.img_size = self.feat_size * self.stride

        def conv_module(in_planes, out_planes, kernel_size=3, padding=1):
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

    def forward(self, x):
        """Forward pass with input x.

        Args:
            x (Tensor): of shape (bs, C, H, W).
        Returns:
            (Tensor): of shape (bs, 4).
        """
        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)

    def get_score_map(self, x):
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

    def soft_argmax(self, score_map):
        """Get soft-argmax coordinate for the given score map.

        Args:
            score_map (self.feat_size, self.feat_size): the last score map
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


@HEADS.register_module()
class ScoreHead(nn.Module):
    """Predict the confidence score of target in current frame.

    Cascade multiple FC layer and empose relu on the output of last layer.

    Args:
        input_dim (int): the dim of input.
        hidden_dim (int): the dim of hidden layers.
        output_dim (int): the dim of output.
        num_layers (int): the number of FC layers.
        BN (bool, optional): whether to use BN after each FC layer.
            Defaults to False.
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 BN=False):
        super(ScoreHead, self).__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(
                nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                for n, k in zip([input_dim] + hidden_dims, hidden_dims +
                                [output_dim]))
        else:
            self.layers = nn.ModuleList(
                nn.Linear(n, k)
                for n, k in zip([input_dim] + hidden_dims, hidden_dims +
                                [output_dim]))

    def forward(self, x):
        """Forward function for `ScoreHead`.

        Args:
            x (Tensor): of shape (bs, C, H, W).

        Returns:
            Tensor: of shape (bs, num_query, 1).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER.register_module()
class StarkTransformer(Transformer):
    """The transformer head used in STARK. `STARK.

    <https://arxiv.org/abs/2103.17154>`_.

    This module follows the official DETR implementation.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(StarkTransformer, self).__init__(
            encoder=encoder, decoder=decoder, init_cfg=init_cfg)

    def forward(self, x, mask, query_embed, pos_embed):
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
            tuple[Tensor]: results of decoder containing the following tensor.
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


@HEADS.register_module()
class StarkHead(BaseModule):
    """STARK head module for bounding box regression and prediction of
    confidence score of tracking bbox.

    This module is proposed in
    "Learning Spatio-Temporal Transformer for Visual Tracking".
    `STARK <https://arxiv.org/abs/2103.17154>`_.

    Args:
        num_query (int): Number of query in transformer.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        run_cls_head (bool, optional): whether to run classification head.
            Defaults to False.
        run_bbox_head (bool, optional): whether to run bboxes regression head.
            Defaults to True.
        bbox_head (obj:`mmcv.ConfigDict`|dict, optional): Config for bbox head.
            Defaults to None.
        cls_head (obj:`mmcv.ConfigDict`|dict, optional): Config for
            classification head. Defaults to None.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the bbox
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the bbox
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
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
                 run_cls_head=False,
                 run_bbox_head=True,
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
                 **kwargs):
        super(StarkHead, self).__init__(init_cfg=init_cfg)
        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.run_bbox_head = run_bbox_head
        self.run_cls_head = run_cls_head
        self.bbox_head = build_head(bbox_head)
        if self.run_bbox_head:
            self.loss_bbox = build_loss(loss_bbox)
            self.loss_iou = build_loss(loss_iou)
        if self.run_cls_head:
            self.cls_head = build_head(cls_head)
            self.loss_cls = build_loss(loss_cls)
        self.embed_dims = self.transformer.embed_dims
        self.num_query = num_query
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def init_weights(self):
        """Parameters initialization."""
        self.transformer.init_weights()

    def forward_bbox_head(self, feat, enc_mem):
        """
        Args:
            feat: output embeddings of decoder, with shape
                (1, bs, num_query, c).
            enc_mem: output embeddings of encoder, with shape
                (feats_flatten_len, bs, C)

                Here, 'feats_flatten_len' = z_feat_h*z_feat_w*2 + \
                    x_feat_h*x_feat_w.
                'z_feat_h' and 'z_feat_w' denote the height and width of the
                template features respectively.
                'x_feat_h' and 'x_feat_w' denote the height and width of search
                features respectively.
        Returns:
            Tensor: of shape (bs, num_query, 4). The bbox is in
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
        outputs_coord = outputs_coord.view(bs, num_query, 4)
        return outputs_coord

    def forward_head(self,
                     feat,
                     enc_mem,
                     run_bbox_head=False,
                     run_cls_head=False):
        """
        Args:
            feat: the output embeddings of decoder, with shape
                (1, bs, num_query, c)
            enc_mem: the output embeddings of encoder, with shape
                (feats_flatten_len, bs, c)

                Here, 'feats_flatten_len' = z_feat_h*z_feat_w*2 + \
                    x_feat_h*x_feat_w.
                'z_feat_h' and 'z_feat_w' denote the height and width of the
                template features respectively.
                'x_feat_h' and 'x_feat_w' denote the height and width of search
                features respectively.
        Returns:
            dict:
                - 'pred_bboxes': of shape (bs, num_query, 4), in
                    [tl_x, tl_y, br_x, br_y] format
                - 'pred_logit': of shape (bs, num_query, 1)
        """
        out_dict = {}
        if run_cls_head:
            # forward the classification head
            out_dict['pred_logits'] = self.cls_head(feat)[-1]
        if run_bbox_head:
            # forward the box prediction head
            out_dict['pred_bboxes'] = self.forward_bbox_head(feat, enc_mem)

        return out_dict

    def forward(self, inputs):
        """"
        Args:
            inputs (dict(Tensor)):
                - 'feat': (Tensor) of shape (bs, c, feats_flatten_len)
                - 'mask': (Tensor) of shape (bs, 1, feats_flatten_len)
                - 'pos_embed': (Tensor) of shape (feats_flatten_len, bs, c)

                Here, 'feats_flatten_len' = z_feat_h*z_feat_w*2 + \
                    x_feat_h*x_feat_w.
                'z_feat_h' and 'z_feat_w' denote the height and width of the
                template features respectively.
                'x_feat_h' and 'x_feat_w' denote the height and width of
                search features respectively.

        Returns:
             (dict):
                - 'pred_bboxes': (Tensor) of shape (bs, num_query, 4), in
                    [tl_x, tl_y, br_x, br_y] format
                - 'pred_logit': (Tensor) of shape (bs, num_query, 1)
        """
        # outs_dec is in (1, bs, num_query, c) shape
        # enc_mem is in (feats_flatten_len, bs, c) shape
        outs_dec, enc_mem = self.transformer(inputs['feat'], inputs['mask'],
                                             self.query_embedding.weight,
                                             inputs['pos_embed'])

        track_results = self.forward_head(
            outs_dec,
            enc_mem,
            run_bbox_head=self.run_bbox_head,
            run_cls_head=self.run_cls_head)
        return track_results

    def reg_loss(self, pred_bboxes, gt_bboxes):
        """"Loss function for bounding box regression.

        Args:
            pred_bboxes (Tensor): predicted bboxes of shape (bs, 4),
                in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (list[Tensor]): ground truth bboxes of shape (bs, 4),
                 in [tl_x, tl_y, br_x, br_y] format.
        Returns:
            dict: a dictionary of regression loss components.
        """
        losses = dict()
        # regression IoU loss, defaultly GIoU loss
        if (pred_bboxes[:, :2] >= pred_bboxes[:, 2:]).any() or (
                gt_bboxes[:, :2] >= gt_bboxes[:, 2:]).any():
            # the first several iterations of forward may return invalid bbox
            # coordinates.
            losses['loss_iou'] = torch.tensor(0.0).to(pred_bboxes)
        else:
            losses['loss_iou'] = self.loss_iou(pred_bboxes, gt_bboxes)
        # regression L1 loss
        losses['loss_bbox'] = self.loss_bbox(pred_bboxes, gt_bboxes)

        return losses

    def cls_loss(self, pred_logits, gt_labels):
        """"Loss function for classification head.

        Args:
            pred_logits (Tensor): prediction after using sigmoid,
                with shape (bs, 1).
            gt_labels (Tensor): ground truth labels,
                with shape (bs, 1).
        Returns:
            dict: a dictionary of classification loss components.
        """
        losses = dict()
        losses['loss_cls'] = self.loss_cls(pred_logits, gt_labels)
        return losses
