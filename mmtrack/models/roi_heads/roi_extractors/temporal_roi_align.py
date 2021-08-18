import torch
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS

from .single_level_roi_extractor import SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class TemporalRoIAlign(SingleRoIExtractor):

    def __init__(self,
                 ms_roi_algin_cfg=dict(sampling_point=4),
                 tafa_cfg=dict(num_attention_blocks=8),
                 *args,
                 **kwargs):
        super(TemporalRoIAlign, self).__init__(*args, **kwargs)
        self.ms_roi_algin_cfg = ms_roi_algin_cfg
        self.tafa_cfg = tafa_cfg
        if self.tafa_cfg:
            self.embed_network = ConvModule(
                self.out_channels,
                self.out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None)

    def tafa(self, x, ref_x):
        # (img_n, roi_n, C, 7, 7)
        x = torch.cat((x, ref_x), dim=0)

        img_n, roi_n, _, roi_h, roi_w = x.size()
        num_attention_blocks = self.tafa_cfg.get('num_attention_blocks', 8)

        # 1. Pass through a tiny embed network
        # (img_n * roi_n, C, 7, 7)
        x_embed = self.embed_network(x.view(img_n * roi_n, -1, roi_h, roi_w))
        c_embed = x_embed.size(1)

        # 2. Perform multi-head attention
        # (img_n, roi_n, num_attention_blocks, C / num_attention_blocks, 7, 7)
        x_embed = x_embed.view(img_n, roi_n, num_attention_blocks, -1, roi_h,
                               roi_w)
        # (1, roi_n, num_attention_blocks, C / num_attention_blocks, 7, 7)
        target_x_embed = x_embed[[0]]
        # (img_n, roi_n, num_attention_blocks, 1, 7, 7)
        ada_weights = torch.sum(
            x_embed * target_x_embed, dim=3, keepdim=True) / (
                float(c_embed / num_attention_blocks)**0.5)
        # (img_n, roi_n, num_attention_blocks, C / num_attention_blocks, 7, 7)
        ada_weights = ada_weights.expand(-1, -1, -1,
                                         int(c_embed / num_attention_blocks),
                                         -1, -1).contiguous()
        # (img_n, roi_n, C, 7, 7)
        ada_weights = ada_weights.view(img_n, roi_n, c_embed, roi_h, roi_w)
        ada_weights = ada_weights.softmax(dim=0)

        # 3. Aggregation
        x = (x * ada_weights).sum(dim=0)
        return x

    def ms_roi_align(self, roi_feats, ref_feats):
        # 1. Compute cos similarity maps.
        # (roi_n, C, 7, 7)
        roi_feats_embed = roi_feats
        # (img_n, C, H, W)
        ref_feats_embed = ref_feats
        roi_feats_embed = roi_feats_embed / roi_feats_embed.norm(
            p=2, dim=1, keepdim=True)
        ref_feats_embed = ref_feats_embed / ref_feats_embed.norm(
            p=2, dim=1, keepdim=True)
        roi_n, c_embed, roi_h, roi_w = roi_feats_embed.size()
        img_n, c_embed, img_h, img_w = ref_feats_embed.size()
        # (roi_n, 7, 7, C)
        roi_feats_embed = roi_feats_embed.permute(0, 2, 3, 1).contiguous()
        # (roi_n * 7 * 7, C)
        roi_feats_embed = roi_feats_embed.view(-1, c_embed)
        # (C, img_n, H, W)
        ref_feats_embed = ref_feats_embed.permute(1, 0, 2, 3).contiguous()
        # (C, img_n * H * W)
        ref_feats_embed = ref_feats_embed.view(c_embed, -1)
        # (roi_n * 7 * 7, img_n * H * W)
        cos_similarity_maps = roi_feats_embed.mm(ref_feats_embed)
        # (roi_n * 7 * 7, img_n, H * W)
        cos_similarity_maps = cos_similarity_maps.view(-1, img_n,
                                                       img_h * img_w)

        # 2. Pick the top K points based on the similarity scores.
        # (roi_n * 7 * 7, img_n, top_k)
        values, indices = cos_similarity_maps.topk(
            k=self.ms_roi_algin_cfg.get('sampling_point', 4),
            dim=2,
            largest=True,
        )
        # (roi_n * 7 * 7, img_n, top_k)
        values = values.softmax(dim=2)

        # 3. Project these top K points into reference feature maps.
        # (H, W, img_n, C)
        ref_feats_reshape = ref_feats.permute(2, 3, 0, 1).contiguous()
        # (H * W, img_n, C)
        ref_feats_reshape = ref_feats_reshape.view(-1, img_n, c_embed)
        # (0, roi_n * 7 * 7, C)
        ref_roi_feats = roi_feats.new_zeros(
            (0, roi_n * roi_h * roi_w, c_embed))
        for i in range(img_n):
            # (roi_n * 7 * 7, top_k, C)
            topk_feats = ref_feats_reshape[indices[:, i], i, :]
            # (roi_n * 7 * 7, top_k, 1)
            topk_weights = values[:, i].unsqueeze(-1)
            # (roi_n * 7 * 7, top_k, C)
            one_ref_roi_feats = topk_feats * topk_weights
            # (1, roi_n * 7 * 7, C)
            one_ref_roi_feats = one_ref_roi_feats.sum(dim=1).unsqueeze(0)
            # (img_n, roi_n * 7 * 7, C)
            ref_roi_feats = torch.cat((ref_roi_feats, one_ref_roi_feats),
                                      dim=0)
        # (img_n, roi_n, 7, 7, C)
        ref_roi_feats = ref_roi_feats.view(img_n, roi_n, roi_h, roi_w, c_embed)
        # (img_n, roi_n, C, 7, 7)
        ref_roi_feats = ref_roi_feats.permute(0, 1, 4, 2, 3)
        return ref_roi_feats

    @force_fp32(apply_to=('feats', 'ref_feats'), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None, ref_feats=None):
        """Forward function."""
        roi_feats = super().forward(feats, rois, roi_scale_factor)

        if ref_feats is None:
            # Directly return roi features for proposals of reference images.
            return roi_feats
        else:
            # We only use the last level of reference feature map to perform
            # Most Similar RoI Align.
            ref_roi_feats = self.ms_roi_align(roi_feats, ref_feats[-1])

            roi_feats = roi_feats.unsqueeze(0)
            if self.tafa_cfg is None:
                temporal_roi_feats = torch.cat((roi_feats, ref_roi_feats),
                                               dim=0)
                temporal_roi_feats = temporal_roi_feats.mean(dim=0)
            else:
                temporal_roi_feats = self.tafa(roi_feats, ref_roi_feats)
            return temporal_roi_feats
