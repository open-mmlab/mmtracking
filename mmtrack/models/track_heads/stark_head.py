import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils import Transformer
from mmdet.models.utils.builder import TRANSFORMER
from torch import nn


def conv(in_planes,
         out_planes,
         kernel_size=3,
         stride=1,
         padding=1,
         dilation=1,
         freeze_bn=False):

    return ConvModule(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        dilation=dilation,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        inplace=True)


class Corner_Predictor(nn.Module):
    """Corner Predictor module."""

    def __init__(self,
                 inplanes=64,
                 channel=256,
                 feat_sz=20,
                 stride=16,
                 freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)
        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)
        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1,
                                                             1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """Forward pass with input x."""
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(
                score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(
                score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br),
                               dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack(
                (coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
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

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """get soft-argmax coordinate for a given heatmap."""
        score_vec = score_map.view(
            (-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(
                nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(
                nn.Linear(n, k)
                for n, k in zip([input_dim] + h, h + [output_dim]))

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
            x (Tensor): Input query with shape [h1w1+h2w2, bs, c] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h1w1+h2w2].
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
                      [h1w1+h2hw, bs, embed_dims].
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

    def __init__(self,
                 num_classes,
                 in_channels,
                 strides=[16],
                 num_query=1,
                 num_cls_fcs=3,
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
                 train_cfg=None,
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        self.num_cls_fcs = num_cls_fcs
        # self.num_query = num_query
        # self.num_classes = num_classes
        # self.in_channels = in_channels
        self.strides = strides
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

        # self.test_cfg = test_cfg
        # self.fp16_enabled = False

        # self.positional_encoding = build_positional_encoding(
        #     positional_encoding)
        # self.transformer = build_transformer(transformer)
        # self.embed_dims = self.transformer.embed_dims

        # self.embed_dims = self.transformer.embed_dims
        # assert 'num_feats' in positional_encoding
        # num_feats = positional_encoding['num_feats']
        # assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
        #     f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
        #     f' and {num_feats}.'

        # self._init_layers()

    def _init_layers(self):
        feat_sz = self.test_cfg['search_size'] // self.strides[0]
        self.bbox_head = Corner_Predictor(
            self.embed_dims,
            self.embed_dims,
            feat_sz=feat_sz,
            stride=self.strides[0])
        self.cls_head = MLP(self.embed_dims, self.embed_dims, 1,
                            self.num_cls_fcs)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def forward_bbox_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (h1w1+h2w2, B, C)"""
        # adjust shape
        feat_len_x = self.bbox_head.feat_sz**2
        enc_opt = memory[-feat_len_x:].transpose(
            0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
        att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.bbox_head.feat_sz,
                            self.bbox_head.feat_sz)
        # run the corner head
        outputs_coord = bbox_xyxy_to_cxcywh(self.bbox_head(opt_feat))
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}
        return out, outputs_coord_new

    def forward_head(self, hs, memory, run_box_head=False, run_cls_head=False):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        out_dict = {}
        if run_cls_head:
            # forward the classification head
            out_dict.update({'pred_logits': self.cls_head(hs)[-1]})
        if run_box_head:
            # forward the box prediction head
            out_dict_box, _ = self.forward_bbox_head(hs, memory)
            # merge results
            out_dict.update(out_dict_box)

        return out_dict

    def forward(self, inputs, run_box_head=True, run_cls_head=True):
        """"Forward function for a single feature level.
        Args:
            feat (Tensor): Input feature from backbone's single stage, shape
                [bs, c, z1_h*z1_w+z2_h*z2_w+x_h*x_w].
            mask (Tensor): shape [bs, 1, z1_h*z1_w+z2_h*z2_w+x_h*x_w]
            pos_embed (Tensor)
        Returns:
             track_results:
                - 'pred_bboxes': shape (bs, Nq, 4)
                - 'pred_logit': shape (bs, Nq, 1)
            outs_dec: [1, bs, num_query, embed_dims]
        """
        # outs_dec: [1, bs, num_query, embed_dims]
        # enc_mem: [h1w1+h2hw, bs, embed_dims]
        outs_dec, enc_mem = self.transformer(inputs['feat'], inputs['mask'],
                                             self.query_embedding.weight,
                                             inputs['pos_embed'])

        track_results = self.forward_head(
            outs_dec,
            enc_mem,
            run_box_head=run_box_head,
            run_cls_head=run_cls_head)
        return track_results, outs_dec
