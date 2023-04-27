# Copyright (c) OpenMMLab. All rights reserved.
from math import ceil
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmdet.structures import SampleList
from mmdet.structures.mask import mask2bbox
from mmdet.utils import ConfigType, InstanceList, OptConfigType
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.registry import MODELS


@MODELS.register_module()
class VITATrackHead(BaseModule):

    def __init__(self,
                 mask_dim: int = 256,
                 enc_window_size: int = 6,
                 use_sim: bool = True,
                 enforce_input_project: bool = True,
                 sim_use_clip: bool = True,
                 num_heads: int = 8,
                 hidden_dim: int = 256,
                 num_queries: int = 100,
                 num_classes: int = 40,
                 num_frame_queries: int = 100,
                 frame_query_encoder: ConfigType = ...,
                 frame_query_decoder: ConfigType = ...,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 *args) -> None:
        super(VITATrackHead, self).__init__()
        self.window_size = enc_window_size
        self.vita_mask_features = Conv2d(
            in_channels=mask_dim,
            out_channels=mask_dim,
            kernel_size=1,
            stride=1,
            bias=True)
        self.frame_query_encoder = MODELS.build(frame_query_encoder)
        self.frame_query_decoder = MODELS.build(frame_query_decoder)
        self.num_heads = num_heads
        self.sim_use_clip = sim_use_clip
        self.use_sim = use_sim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_transformer_decoder_layers = frame_query_decoder.num_layers

        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.fq_pos = nn.Embedding(num_frame_queries, hidden_dim)

        self.cls_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))

        if self.use_sim:
            self.sim_embed_frame = nn.Linear(hidden_dim, hidden_dim)
            if self.sim_use_clip:
                self.sim_embed_clip = nn.Linear(hidden_dim, hidden_dim)

        if enforce_input_project:
            self.input_proj_dec = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.input_proj_dec = nn.Sequential()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(
        self,
        x: Tuple[Tensor],
        data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the track head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # forward
        all_cls_scores, all_mask_preds = self(x, data_samples)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances)
        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def forward(self, frame_queries: Tensor) -> Tuple[Tensor, ...]:
        """Forward function.

        L: Number of Layers.
        B: Batch size.
        T: Temporal window size. Number of frames per video.
        C: Channel size.
        fQ: Number of frame-wise queries from IFC.
        cQ: Number of clip-wise queries to decode Q.
        """
        L, BT, fQ, C = frame_queries.shape
        B = BT // self.num_frames if self.training else 1
        T = self.num_frames if self.training else BT // B

        frame_queries = frame_queries.reshape(L * B, T, fQ, C)
        frame_queries = frame_queries.permute(1, 2, 0, 3).contiguous()
        frame_queries = self.input_proj_dec(frame_queries)

        # for window attention
        if self.window_size > 0:
            pad = int(ceil(T / self.window_size)) * self.window_size - T
            _T = pad + T
            frame_queries = F.pad(frame_queries, (0, 0, 0, 0, 0, 0, 0, pad))
            enc_mask = frame_queries.new_ones(L * B, _T).bool()
            enc_mask[:, :T] = False
        else:
            enc_mask = None

        frame_queries = self.encode_frame_query(frame_queries, enc_mask)
        # (LB, T*fQ, C)
        frame_queries = frame_queries[:T].flatten(0, 1)

        if self.use_sim:
            fq_embed = self.sim_embed_frame(frame_queries)
            fq_embed = fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
        else:
            fq_embed = None

        dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L * B,
                                                              1).flatten(0, 1)

        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, L * B, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, L * B, 1))

        decoder_outputs = []
        for i in range(self.num_transformer_decoder_layers):
            # cross_attn + self_attn
            layer = self.frame_query_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=frame_queries,
                value=frame_queries,
                query_pos=query_embed,
                key_pos=dec_pos,
                attn_masks=None,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            if self.training or (i == self.num_transformer_decoder_layers - 1):
                decoder_out = self.frame_query_decoder.post_norm(query_feat)
                decoder_out = decoder_out.transpose(0, 1)
                decoder_outputs.append(
                    decoder_out.view(L, B, self.num_queries, C))

        decoder_outputs = torch.stack(decoder_outputs, dim=0)

        all_cls_pred = self.cls_embed(decoder_outputs)
        all_mask_embed = self.mask_embed(decoder_outputs)
        if self.use_sim and self.sim_use_clip:
            all_cq_embed = self.sim_embed_clip(decoder_outputs)
        else:
            all_cq_embed = [None] * self.num_transformer_decoder_layers

        return all_cls_pred, all_mask_embed, all_cq_embed, fq_embed

    def encode_frame_query(self, frame_queries, attn_mask):
        # Not using window-based attention if self.window_size == 0.
        if self.window_size == 0:
            return_shape = frame_queries.shape
            # (T, fQ, LB, C) -> (T*fQ, LB, C)
            frame_queries = frame_queries.flatten(0, 1)

            # TODO: add
            frame_queries = frame_queries.view(return_shape)
            return frame_queries
        # Using window-based attention if self.window_size > 0.
        else:
            T, fQ, LB, C = frame_queries.shape
            win_s = self.window_size
            num_win = T // win_s
            half_win_s = int(ceil(win_s / 2))

            window_mask = attn_mask.view(LB * num_win,
                                         win_s)[...,
                                                None].repeat(1, 1,
                                                             fQ).flatten(1)

            _attn_mask = torch.roll(attn_mask, half_win_s, 1)
            _attn_mask = _attn_mask.view(LB, num_win,
                                         win_s)[...,
                                                None].repeat(1, 1, 1, win_s)
            _attn_mask[:, 0] = _attn_mask[:, 0] | _attn_mask[:, 0].transpose(
                -2, -1)
            _attn_mask[:,
                       -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(
                           -2, -1)
            _attn_mask[:, 0, :half_win_s, half_win_s:] = True
            _attn_mask[:, 0, half_win_s:, :half_win_s] = True
            _attn_mask = _attn_mask.view(
                LB * num_win, 1, win_s, 1, win_s,
                1).repeat(1, self.num_heads, 1, fQ, 1,
                          fQ).view(LB * num_win * self.num_heads, win_s * fQ,
                                   win_s * fQ)
            shift_window_mask = _attn_mask.float() * -1000

            for layer_idx in range(self.frame_query_encoder.num_layers):
                if self.training or layer_idx % 2 == 0:
                    frame_queries = self._window_attn(frame_queries,
                                                      window_mask, layer_idx)
                else:
                    frame_queries = self._shift_window_attn(
                        frame_queries, shift_window_mask, layer_idx)
            return frame_queries

    def _window_attn(self, frame_queries, attn_mask, layer_idx):
        T, fQ, LB, C = frame_queries.shape

        win_s = self.window_size
        num_win = T // win_s

        frame_queries = frame_queries.view(num_win, win_s, fQ, LB, C)
        frame_queries = frame_queries.permute(1, 2, 3, 0, 4).reshape(
            win_s * fQ, LB * num_win, C)

        frame_queries = self.frame_query_encoder.layers[layer_idx](
            frame_queries, query_key_padding_mask=attn_mask)

        frame_queries = frame_queries.reshape(win_s, fQ, LB, num_win,
                                              C).permute(3, 0, 1, 2,
                                                         4).reshape(
                                                             T, fQ, LB, C)
        return frame_queries

    def _shift_window_attn(self, frame_queries, attn_mask, layer_idx):
        T, fQ, LB, C = frame_queries.shape

        win_s = self.window_size
        num_win = T // win_s
        half_win_s = int(ceil(win_s / 2))

        frame_queries = torch.roll(frame_queries, half_win_s, 0)
        frame_queries = frame_queries.view(num_win, win_s, fQ, LB, C)
        frame_queries = frame_queries.permute(1, 2, 3, 0, 4).reshape(
            win_s * fQ, LB * num_win, C)

        frame_queries = self.frame_query_encoder.layers[layer_idx](
            frame_queries, attn_masks=attn_mask)
        frame_queries = frame_queries.reshape(win_s, fQ, LB, num_win,
                                              C).permute(3, 0, 1, 2,
                                                         4).reshape(
                                                             T, fQ, LB, C)

        frame_queries = torch.roll(frame_queries, -half_win_s, 0)

        return frame_queries

    def predict(self,
                mask_features: Tensor,
                frame_queries: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        all_cls_pred, all_mask_embed, _, _ = self(frame_queries)

        results = self.predict_by_feat(
            all_cls_pred,
            all_mask_embed,
            mask_features,
            batch_img_metas=batch_img_metas,
            rescale=rescale)
        return results

    def predict_by_feat(self,
                        all_cls_pred: Tensor,
                        all_mask_embed: Tensor,
                        mask_features: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = True) -> InstanceList:

        cls_pred = all_cls_pred[-1, -1, 0]
        mask_embed = all_mask_embed[-1, -1, 0]
        # The input is a video, and a batch is a video,
        # so the img shape of each image is the same.
        # Here is the first image.
        img_meta = batch_img_metas[0]

        scores = F.softmax(cls_pred, dim=-1)[:, :-1]

        max_per_image = self.test_cfg.get('max_per_video', 10)
        test_interpolate_chunk_size = self.test_cfg.get(
            'test_interpolate_chunk_size', 5)
        scores_per_video, topk_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels = torch.arange(
            self.num_classes,
            device=cls_pred.device).unsqueeze(0).repeat(self.num_queries,
                                                        1).flatten(0, 1)
        labels_per_video = labels[topk_indices]

        query_indices = topk_indices // self.num_classes
        mask_embed = mask_embed[query_indices]

        masks_per_video = []
        numerator = torch.zeros(
            len(mask_embed), dtype=torch.float, device=cls_pred.device)
        denominator = torch.zeros(
            len(mask_embed), dtype=torch.float, device=cls_pred.device)
        for i in range(ceil(len(mask_features) / test_interpolate_chunk_size)):
            mask_feat = mask_features[i * test_interpolate_chunk_size:(i + 1) *
                                      test_interpolate_chunk_size].to(
                                          cls_pred.device)

            mask_pred = torch.einsum('qc,tchw->qthw', mask_embed, mask_feat)

            pad_height, pad_width = img_meta['pad_shape']
            rz_height, rz_width = img_meta['img_shape']
            # upsample masks
            mask_pred = F.interpolate(
                mask_pred,
                size=(pad_height, pad_width),
                mode='bilinear',
                align_corners=False)

            # crop the padding area
            mask_pred = mask_pred[:, :, :rz_height, :rz_width]
            ori_height, ori_width = img_meta['ori_shape']

            interim_mask_soft = mask_pred.sigmoid()
            interim_mask_hard = interim_mask_soft > 0.5

            numerator += (interim_mask_soft.flatten(1) *
                          interim_mask_hard.flatten(1)).sum(1)
            denominator += interim_mask_hard.flatten(1).sum(1)

            mask_pred = F.interpolate(
                mask_pred,
                size=(ori_height, ori_width),
                mode='bilinear',
                align_corners=False) > 0.

            masks_per_video.append(mask_pred)

        masks_per_video = torch.cat(masks_per_video, dim=1)
        scores_per_video *= (numerator / (denominator + 1e-6))

        # format top-10 predictions
        results = []
        for img_idx in range(len(batch_img_metas)):
            pred_track_instances = InstanceData()

            pred_track_instances.masks = masks_per_video[:, img_idx]
            pred_track_instances.bboxes = mask2bbox(masks_per_video[:,
                                                                    img_idx])
            pred_track_instances.labels = labels_per_video
            pred_track_instances.scores = scores_per_video
            pred_track_instances.instances_id = torch.arange(10)

            results.append(pred_track_instances)

        return results
