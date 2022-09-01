# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import List, Optional, Tuple

from mmdet.models.task_modules import SamplingResult
from mmdet.structures.bbox import bbox2roi
from mmengine.model import BaseModule
from torch import Tensor

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import InstanceList, SampleList


@MODELS.register_module()
class RoITrackHead(BaseModule, metaclass=ABCMeta):
    """The roi track head.

    This module is used in multi-object tracking methods, such as MaskTrack
    R-CNN.

    Args:
        roi_extractor (dict): Configuration of roi extractor. Defaults to None.
        embed_head (dict): Configuration of embed head. Defaults to None.
        train_cfg (dict): Configuration when training. Defaults to None.
        test_cfg (dict): Configuration when testing. Defaults to None.
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 roi_extractor: Optional[dict] = None,
                 embed_head: Optional[dict] = None,
                 regress_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 *args,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if embed_head is not None:
            self.init_embed_head(roi_extractor, embed_head)

        if regress_head is not None:
            raise NotImplementedError('Regression head is not supported yet.')

        self.init_assigner_sampler()

    def init_embed_head(self, roi_extractor, embed_head) -> None:
        """Initialize ``embed_head``"""
        self.roi_extractor = MODELS.build(roi_extractor)
        self.embed_head = MODELS.build(embed_head)

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    @property
    def with_track(self) -> bool:
        """bool: whether the multi-object tracker has an embed head"""
        return hasattr(self, 'embed_head') and self.embed_head is not None

    def extract_roi_feats(
            self, feats: List[Tensor],
            bboxes: List[Tensor]) -> Tuple[Tuple[Tensor], List[int]]:
        """Extract roi features.

        Args:
            feats (list[Tensor]): list of multi-level image features.
            bboxes (list[Tensor]): list of bboxes in sampling result.

        Returns:
            tuple[tuple[Tensor], list[int]]: The extracted roi features and
            the number of bboxes in each image.
        """
        rois = bbox2roi(bboxes)
        bbox_feats = self.roi_extractor(feats[:self.roi_extractor.num_inputs],
                                        rois)
        num_bbox_per_img = [len(bbox) for bbox in bboxes]
        return bbox_feats, num_bbox_per_img

    def loss(self, key_feats: List[Tensor], ref_feats: List[Tensor],
             rpn_results_list: InstanceList, data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            key_feats (list[Tensor]): list of multi-level image features.
            ref_feats (list[Tensor]): list of multi-level ref_img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        assert len(rpn_results_list) == len(data_samples)
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        for data_sample in data_samples:
            batch_gt_instances.append(data_sample.gt_instances)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)

        if self.with_track:
            num_imgs = len(data_samples)
            if batch_gt_instances_ignore is None:
                batch_gt_instances_ignore = [None] * num_imgs
            sampling_results = []
            for i in range(num_imgs):
                rpn_results = rpn_results_list[i]

                assign_result = self.bbox_assigner.assign(
                    rpn_results, batch_gt_instances[i],
                    batch_gt_instances_ignore[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    rpn_results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in key_feats])
                sampling_results.append(sampling_result)

        losses = dict()

        if self.with_track:
            track_results = self.track_loss(key_feats, ref_feats,
                                            sampling_results, data_samples)
            losses.update(track_results['loss_track'])

        return losses

    def track_loss(self, key_feats: List[Tensor], ref_feats: List[Tensor],
                   sampling_results: List[SamplingResult],
                   data_samples: SampleList, **kwargs) -> dict:
        """Run forward function and calculate loss for track head in training.

        Args:
            key_feats (list[Tensor]): list of multi-level image features.
            ref_feats (list[Tensor]): list of multi-level ref_img features.
            sampling_results (list[:obj:`SamplingResult`]): List of Bbox
                sampling result.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        bboxes = [res.bboxes for res in sampling_results]
        bbox_feats, num_bbox_per_img = self.extract_roi_feats(
            key_feats, bboxes)
        # batch_size is 1
        ref_gt_bboxes = [data_samples[0].ref_gt_instances.bboxes]
        ref_bbox_feats, num_bbox_per_ref_img = self.extract_roi_feats(
            ref_feats, ref_gt_bboxes)

        gt_instance_ids = [data_samples[0].gt_instances.instances_id]
        ref_gt_instance_ids = [data_samples[0].ref_gt_instances.instances_id]

        loss_track = self.embed_head.loss(bbox_feats, ref_bbox_feats,
                                          num_bbox_per_img,
                                          num_bbox_per_ref_img,
                                          sampling_results, gt_instance_ids,
                                          ref_gt_instance_ids)
        track_results = dict(loss_track=loss_track)

        return track_results

    def predict(self, roi_feats: Tensor,
                prev_roi_feats: Tensor) -> List[Tensor]:
        """Perform forward propagation of the tracking head and predict
        tracking results on the features of the upstream network.

        Args:
            roi_feats (Tensor): Feature map of current images rois.
            prev_roi_feats (Tensor): Feature map of previous images rois.

        Returns:
            list[Tensor]: The predicted similarity_logits of each pair of key
            image and reference image.
        """
        return self.embed_head.predict(roi_feats, prev_roi_feats)[0]
