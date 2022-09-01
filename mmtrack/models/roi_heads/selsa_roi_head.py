# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from mmdet.models import StandardRoIHead
from mmdet.structures.bbox import bbox2roi
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import ConfigType, InstanceList, SampleList


@MODELS.register_module()
class SelsaRoIHead(StandardRoIHead):
    """selsa roi head."""

    def loss(self, x: Tuple[Tensor], ref_x: Tuple[Tensor],
             rpn_results_list: InstanceList,
             ref_rpn_results_list: InstanceList,
             data_samples: SampleList) -> dict:
        """
        Args:
            x (Tuple[Tensor]): list of multi-level img features.
            ref_x (Tuple[Tensor]): list of multi-level ref_img features.
            rpn_results_list (InstanceList): list of region proposals.
            ref_rpn_results_list (InstanceList): list of region proposals
                from reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` and 'metainfo'.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
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

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(data_samples)
            sampling_results = []
            for i in range(num_imgs):
                # rename rpn_results.bboxes to rpn_results.priors
                rpn_results = rpn_results_list[i]
                rpn_results.priors = rpn_results.pop('bboxes')
                assign_result = self.bbox_assigner.assign(
                    rpn_results, batch_gt_instances[i],
                    batch_gt_instances_ignore[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    rpn_results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, ref_x, sampling_results,
                                          ref_rpn_results_list)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x: Tuple[Tensor], ref_x: Tuple[Tensor],
                      rois: Tensor, ref_rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (Tuple[Tensor]): List of multi-level img features.
            ref_x (Tuple[Tensor]): List of multi-level reference img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            ref_rois (Tensor): Reference RoIs with the shape (n, 5) where the
                first column indicates batch id of each RoI.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """

        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            ref_feats=ref_x[:self.bbox_roi_extractor.num_inputs])

        ref_bbox_feats = self.bbox_roi_extractor(
            ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            ref_bbox_feats = self.shared_head(ref_bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, ref_bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor], ref_x: Tuple[Tensor],
                  sampling_results: InstanceList,
                  ref_rpn_results_list: InstanceList):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (Tuple[Tensor]): list of multi-level img features.
            ref_x (Tuple[Tensor]): list of multi-level ref_img features.
            sampling_results (InstanceList): Sampleing results.
            ref_rpn_results_list (InstanceList): list of region proposals
                from reference images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        rois = bbox2roi([res.bboxes for res in sampling_results])
        ref_rois = bbox2roi([res.bboxes for res in ref_rpn_results_list])
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def predict(self,
                x: Tuple[Tensor],
                ref_x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                ref_rpn_results_list: InstanceList,
                data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (Tuple[Tensor]): All scale level feature maps of images.
            ref_x (Tuple[Tensor]): All scale level feature maps of reference
                mages.
            rpn_results_list (InstanceList): list of region proposals.
            ref_rpn_results_list (InstanceList): list of region
                proposals from reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` and 'metainfo'.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        batch_img_metas = [
            data_samples.metainfo for data_samples in data_samples
        ]

        results_list = self.predict_bbox(
            x,
            ref_x,
            rpn_results_list,
            ref_rpn_results_list,
            batch_img_metas,
            self.test_cfg,
            rescale=rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale)

        return results_list

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     ref_x: Tuple[Tensor],
                     rpn_results_list: InstanceList,
                     ref_rpn_results_list: InstanceList,
                     batch_img_metas: List[dict],
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (Tuple[Tensor]): All scale level feature maps of images.
            ref_x (Tuple[Tensor]): All scale level feature maps of reference
                mages.
            rpn_results_list (InstanceList): List of region proposals.
            ref_rpn_results_list (InstanceList): List of region
                proposals from reference images.
            batch_img_metas (List[dict]): _List of image information.
            rcnn_test_cfg (ConfigType): `test_cfg` of R-CNN.
            rescale (bool, optional): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        rois = bbox2roi([res.bboxes for res in rpn_results_list])
        ref_rois = bbox2roi([res.bboxes for res in ref_rpn_results_list])
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in rpn_results_list)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        bbox_pred = bbox_pred.split(
            num_proposals_per_img,
            0) if bbox_pred is not None else [None, None]

        # apply bbox post-processing to each image individually
        result_list = self.bbox_head.predict_by_feat(
            rois,
            cls_score,
            bbox_pred,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)

        return result_list
