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
    """Corner Predictor module.

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

        # about coordinates and indexes
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_size).view(-1,
                                                               1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_size, 1)) \
                .view((self.feat_size * self.feat_size,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_size)) \
                .view((self.feat_size * self.feat_size,)).float().cuda()

    def forward(self, x):
        """Forward pass with input x.

        Args:
            x (Tensor): of shape [bs, C, H, W].
        Returns:
            (Tensor): of shape [bs,4]
        """
        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)

    def get_score_map(self, x):
        """score map branch.

        Args:
            x (Tensor): of shape [bs, C, H, W].
        Returns:
            score_map_tl (Tensor[bs, 1, H, W]): the score map of top left
                corner of tracking bbox
            score_map_br (Tensor[bs, 1, H, W]): the score map of bottom right
                corner of tracking bbox
        """
        score_map_tl = self.tl_corner_pred(x)
        score_map_br = self.br_corner_pred(x)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map):
        """Get soft-argmax coordinate for a given heatmap.

        Args:
            score_map (self.feat_size, self.feat_size): the last score map
                in bbox_head branch

        Returns:
            exp_x (Tensor): of shape (bs, 1), the value is in range
                [0, self.feat_size * self.stride]
            exp_y (Tensor): of shape (bs, 1), the value is in range
                [0, self.feat_size * self.stride]
        """

        score_vec = score_map.view(
            (-1,
             self.feat_size * self.feat_size))  # (B, feat_size * feat_size)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return exp_x, exp_y


@HEADS.register_module()
class ScoreHead(nn.Module):
    """Predict the confidence score of target in current frame Cascade multiple
    Linear layer and empose relu on the output of last layer.

    Returns:
        Tensor: of shape [bs, 1]
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 BN=False):
        super().__init__()
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
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER.register_module()
class StarkTransformer(Transformer):

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(StarkTransformer, self).__init__(
            encoder=encoder, decoder=decoder, init_cfg=init_cfg)

    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `StarkTransformer`.
        Args:
            x (Tensor): Input query with shape [z_h*z_w*2 + x_h*x_w, bs, c]
                where c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, z_h*z_w*2 + x_h*x_w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [z_h*z_w*2 + x_h*x_w, bs, embed_dims].
        """
        _, bs, _ = x.shape
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]

        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask)
        out_dec = out_dec.transpose(1, 2)
        return out_dec, memory


@HEADS.register_module()
class StarkHead(BaseModule):
    """
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_cls_fcs (int, optional): Number of fully-connected layers used in
            score head. Default to 3.
        stride (int): The stride of the input feature map. Default to 16.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
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
        self.bbox_head = build_head(bbox_head)
        self.cls_head = build_head(cls_head)
        self.embed_dims = self.transformer.embed_dims
        self.num_query = num_query
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

    def init_weights(self):
        self.transformer.init_weights()
        # self.bbox_head.init_weights()
        # self.cls_head.init_weights()

    def forward_bbox_head(self, feat, memory):
        """
        Args:
            feat: output embeddings (1, bs, N, C)
            memory: encoder embeddings (z_h*z_w*2 + x_h*x_w, bs, C)
        Returns:
            Tensor: of shape (bs, Nq, 4), Nq is the number of query in
                transformer. the bbox format is [tl_x, tl_y, br_x, br_y]
        """
        # adjust shape
        feat_len_x = self.bbox_head.feat_size**2
        enc_opt = memory[-feat_len_x:].transpose(
            0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt = feat.squeeze(0).transpose(1, 2)  # (B, C, N)
        att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.bbox_head.feat_size,
                            self.bbox_head.feat_size)
        # run the corner prediction head
        outputs_coord = self.bbox_head(opt_feat)
        outputs_coord = outputs_coord.view(bs, Nq, 4)
        return outputs_coord

    def forward_head(self,
                     feat,
                     memory,
                     run_box_head=False,
                     run_cls_head=False):
        """
        Args:
            feat: output embeddings (1, bs, N, C)
            memory: encoder embeddings (z_h*z_w*2 + x_h*x_w, bs, C)
        Returns:
            (dict):
                - 'pred_bboxes': shape (bs, Nq, 4), in [tl_x, tl_y, br_x, br_y]
                    format
                - 'pred_logit': shape (bs, Nq, 1)
        """
        out_dict = {}
        if run_cls_head:
            # forward the classification head
            out_dict['pred_logits'] = self.cls_head(feat)[-1]
        if run_box_head:
            # forward the box prediction head
            out_dict['pred_bboxes'] = self.forward_bbox_head(feat, memory)

        return out_dict

    def forward(self, inputs, run_box_head=True, run_cls_head=True):
        """"
        Args:
            inputs (Tensor): Input feature from backbone's single stage, shape
                'feat': (Tensor) of shape (bs, c, z_h*z_w*2 + x_h*x_w)
                'mask': (Tensor) of shape (bs, 1, z_h*z_w*2 + x_h*x_w)
                'pos_embed': (Tensor) of shape (Nq, c)
        Returns:
             track_results:
                - 'pred_bboxes': (Tensor) of shape (bs, Nq, 4), in
                    [cx, cy, w, h] format
                - 'pred_logit': (Tensor) of shape (bs, Nq, 1)
            outs_dec: [1, bs, num_query, embed_dims]
        """
        # outs_dec: [1, bs, num_query, embed_dims]
        # enc_mem: [z_h*z_w*2 + x_h*x_w, bs, embed_dims]
        outs_dec, enc_mem = self.transformer(inputs['feat'], inputs['mask'],
                                             self.query_embedding.weight,
                                             inputs['pos_embed'])

        track_results = self.forward_head(
            outs_dec,
            enc_mem,
            run_box_head=run_box_head,
            run_cls_head=run_cls_head)
        return track_results, outs_dec

    def loss(self, pred_bboxes, gt_bboxes):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
        """

        losses = dict()

        # regression IoU loss, defaultly GIoU loss
        if (pred_bboxes[:, :2] >= pred_bboxes[:, 2:]).any() or (
                gt_bboxes[:, :2] >= gt_bboxes[:, 2:]).any():
            losses['loss_iou'] = torch.tensor(0.0).to(pred_bboxes)
        else:
            losses['loss_iou'] = self.loss_iou(pred_bboxes, gt_bboxes)
        # regression L1 loss
        losses['loss_bbox'] = self.loss_bbox(pred_bboxes, gt_bboxes)
        return losses
