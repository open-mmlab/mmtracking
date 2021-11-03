import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils import Transformer
from mmdet.models.utils.builder import TRANSFORMER
from torch import nn

@HEADS.register_module()
class CornerPredictorHead(nn.Module):
    """Corner Predictor module.

    Args:
        inplanes (int): input channle
        channel (int): the output channel of the first conv block
        feat_size (int): the size of feature map
        stride (int): the stride of feature map from the backbone
    """

    def __init__(self,
                 inplanes=64,
                 channel=256,
                 feat_size=20,
                 stride=16):
        super(CornerPredictorHead, self).__init__()
        self.feat_size = feat_size
        self.stride = stride
        self.img_size = self.feat_size * self.stride

        def conv_module(in_planes,
            out_planes,
            kernel_size=3,
            padding=1):
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
        self.conv1_tl = conv_module(inplanes, channel)
        self.conv2_tl = conv_module(channel, channel // 2)
        self.conv3_tl = conv_module(channel // 2, channel // 4)
        self.conv4_tl = conv_module(channel // 4, channel // 8)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)
        # bottom-right corner
        self.conv1_br = conv_module(inplanes, channel)
        self.conv2_br = conv_module(channel, channel // 2)
        self.conv3_br = conv_module(channel // 2, channel // 4)
        self.conv4_br = conv_module(channel // 4, channel // 8)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)
        # about coordinates and indexs
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
        """score map branch

        Args:
            x (Tensor): of shape [bs, C, H, W].
        Returns:
            score_map_tl (Tensor[bs, 1, H, W]): the score map of top left corner of tracking bbox
            score_map_br (Tensor[bs, 1, H, W]): the score map of bottom right corner of tracking bbox
        """
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map):
        """Get soft-argmax coordinate for a given heatmap.

        Args:
            score_map (self.feat_size, self.feat_size): the last score map in bbox_head branch

        Returns:
            exp_x (Tensor): of shape (bs, 1), the value is in range [0, self.feat_size * self.stride]
            exp_y (Tensor): of shape (bs, 1), the value is in range [0, self.feat_size * self.stride]
        """
    
        score_vec = score_map.view(
            (-1, self.feat_size * self.feat_size))  # (B, feat_size * feat_size)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return exp_x, exp_y


class ScoreHead(nn.Module):
    """Predict the confidence score of target in current frame
    Cascade multiple Linear layer and empose relu on the output of last layer

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
                for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]))
        else:
            self.layers = nn.ModuleList(
                nn.Linear(n, k)
                for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]))

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
            x (Tensor): Input query with shape [z_h*z_w*2 + x_h*x_w, bs, c] where
                c = embed_dims.
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
class StarkHead(DETRHead):
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
                 num_classes,
                 in_channels,
                 num_query=1,
                 num_cls_fcs=3,
                 stride=16,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                 ),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        self.num_cls_fcs = num_cls_fcs
        self.stride = stride
        super(StarkHead, self).__init__(
            num_classes,
            in_channels,
            num_query=num_query,
            transformer=transformer,
            positional_encoding=positional_encoding,
            loss_cls=loss_cls,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )

    def _init_layers(self):
        feat_size = self.test_cfg['search_size'] // self.stride
        self.bbox_head = CornerPredictorHead(
            self.embed_dims,
            self.embed_dims,
            feat_size=feat_size,
            stride=self.stride)
        self.cls_head = ScoreHead(self.embed_dims, self.embed_dims, 1,
                            self.num_cls_fcs)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def forward_bbox_head(self, feat, memory):
        """
        Args:
            feat: output embeddings (1, bs, N, C)
            memory: encoder embeddings (z_h*z_w*2 + x_h*x_w, bs, C)
        Returns:
            Tensor: of shape (bs, Nq, 4), Nq is the number of query in transformer.
                the bbox format is [tl_x, tl_y, br_x, br_y]
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

    def forward_head(self, feat, memory, run_box_head=False, run_cls_head=False):
        """
        Args:
            feat: output embeddings (1, bs, N, C)
            memory: encoder embeddings (z_h*z_w*2 + x_h*x_w, bs, C)
        Returns:
            (dict):
                - 'pred_bboxes': shape (bs, Nq, 4), in [tl_x, tl_y, br_x, br_y] format
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
                - 'pred_bboxes': (Tensor) of shape (bs, Nq, 4), in [cx, cy, w, h] format
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

    def get_bbox(self, pred_bboxes, prev_bbox, resize_factor):
        """This function map the `prediction bboxes` from resized croped image to original image.
        The coordinate origins of them are both the top left corner.

        Args:
            pred_bboxes (Tensor): of shape (B, Nq, 4), in [tl_x, tl_y, br_x, br_y] format.
            prev_bbox (Tensor): of shape (B, 4), in [cx, cy, w, h] format.
            resize_factor (float):

        Returns:
            (Tensor): in [tl_x, tl_y, br_x, br_y] format
        """
        # based in resized croped image
        pred_bboxes = pred_bboxes.view(-1, 4)
        # based in original croped image
        pred_bbox = pred_bboxes.mean(dim=0) / resize_factor  # (cx, cy, w, h)

        # map the bbox to original image
        half_crop_img_size = 0.5 * self.test_cfg['search_size'] / resize_factor
        x_shift, y_shift = prev_bbox[0] - half_crop_img_size, prev_bbox[1] - half_crop_img_size
        pred_bbox[0] += x_shift
        pred_bbox[1] += y_shift
        pred_bbox[2] += x_shift
        pred_bbox[3] += y_shift

        return pred_bbox
