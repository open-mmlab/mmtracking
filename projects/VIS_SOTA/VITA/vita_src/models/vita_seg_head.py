# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmdet.models.dense_heads import Mask2FormerHead as MMDET_Mask2FormerHead
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmtrack.registry import MODELS
from mmtrack.utils import InstanceList


@MODELS.register_module()
class VITASegHead(MMDET_Mask2FormerHead):

    def __init__(self, num_frames: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_frames = num_frames

    def preprocess_gt(self, batch_gt_instances: InstanceList) -> InstanceList:
        """Preprocess the ground truth for all images."""
        final_batch_gt_instances = []
        for gt_instances in batch_gt_instances:
            _device = gt_instances.labels.device
            gt_instances.masks = gt_instances.masks.to_tensor(
                dtype=torch.bool, device=_device)
            # a list used to record which image each instance belongs to
            map_info = gt_instances.map_instances_to_img_idx
            for frame_id in range(self.num_frames):
                ins_index = (map_info == frame_id)
                per_frame_gt = gt_instances[ins_index]
                tmp_instances = InstanceData(
                    labels=per_frame_gt.labels,
                    masks=per_frame_gt.masks.long())
                final_batch_gt_instances.append(tmp_instances)

        return final_batch_gt_instances

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor]]:
        """Forward function.

        Overwriting here is mainly for VITA.
        """
        batch_size = x[0].size(0)
        mask_features, clip_mask_features, multi_scale_memorys = \
            self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        frame_query_list = []
        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask, frame_query = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask, frame_query = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            frame_query_list.append(frame_query)
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        # we only need frame query for VITA
        return cls_pred_list, mask_pred_list, \
            frame_query_list, clip_mask_features

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask, decoder_out

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        for data_sample in batch_data_samples:
            for _ in range(self.num_frames):
                batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # fix batch_img_metas, these keys will be used in self.loss_by_feat
        key_list = [
            'batch_input_shape', 'pad_shape', 'img_shape', 'scale_factor'
        ]
        for key in key_list:
            for batch_id in range(len(batch_img_metas)):
                batch_img_metas[batch_id][key] = batch_img_metas[batch_id][
                    key][0]

        # forward
        all_cls_scores, all_mask_preds, all_frame_queries, mask_features = \
            self(x)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor]) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        all_frame_queries, mask_features = self(x)

        frame_queries = all_frame_queries[-1]

        return frame_queries, mask_features
