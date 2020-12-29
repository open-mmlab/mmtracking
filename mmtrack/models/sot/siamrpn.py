import torch
from addict import Dict
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.models.builder import build_backbone, build_head, build_neck
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from ..builder import MODELS
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class SiamRPN(BaseSingleObjectTracker):
    """SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks.

    This single object tracker is the implementation of `SiamRPN++
    <https://arxiv.org/abs/1812.11703>`_.
    """

    def __init__(self,
                 pretrains=None,
                 backbone=None,
                 neck=None,
                 head=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SiamRPN, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        head = head.copy()
        head.update(train_cfg=train_cfg.rpn, test_cfg=test_cfg.rpn)
        self.head = build_head(head)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.init_weights(pretrains)
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self, pretrain):
        """Initialize the weights of modules in single object tracker.

        Args:
            pretrained (dict): Path to pre-trained weights.
        """
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if self.with_backbone and pretrain.get('backbone', False):
            self.init_module('backbone', pretrain['backbone'])

        if self.with_neck:
            for m in self.neck.modules():
                if isinstance(m, _ConvNd) or isinstance(m, _BatchNorm):
                    m.reset_parameters()

        if self.with_head:
            for m in self.head.modules():
                if isinstance(m, _ConvNd) or isinstance(m, _BatchNorm):
                    m.reset_parameters()

    def forward_template(self, z_img):
        """Extract the features of exemplar images.

        Args:
            z_img (Tensor): of shape (N, C, H, W) encoding input exemplar
                images. Typically H and W equal to 127.

        Returns:
            tuple(Tensor): Multi level feature map of exemplar images.
        """
        z_feat = self.backbone(z_img)
        if self.with_neck:
            z_feat = self.neck(z_feat)

        z_feat_center = []
        for i in range(len(z_feat)):
            left = (z_feat[i].size(3) - self.test_cfg.center_size) // 2
            right = left + self.test_cfg.center_size
            z_feat_center.append(z_feat[i][:, :, left:right, left:right])
        return tuple(z_feat_center)

    def forward_search(self, x_img):
        """Extract the features of search images.

        Args:
            x_img (Tensor): of shape (N, C, H, W) encoding input search
                images. Typically H and W equal to 255.

        Returns:
            tuple(Tensor): Multi level feature map of search images.
        """
        x_feat = self.backbone(x_img)
        if self.with_neck:
            x_feat = self.neck(x_feat)
        return x_feat

    def get_cropped_img(self, img, center_xy, target_size, crop_size,
                        avg_channel):
        """Crop image.

        Only used during testing.

        This function mainly contains two steps:
        1. Crop `img` based on center `center_xy` and size `crop_size`. If the
        cropped image is out of boundary of `img`, use `avg_channel` to pad.
        2. Resize the cropped image to `target_size`.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            center_xy (Tensor): of shape (2, ) denoting the center point for
                cropping image.
            target_size (int): The output size of cropped image.
            crop_size (Tensor): The size for cropping image.
            avg_channel (Tensor): of shape (3, ) denoting the padding values.

        Returns:
            Tensor: of shape (1, C, target_size, target_size) encoding the
            resized cropped image.
        """
        N, C, H, W = img.shape
        context_xmin = int(center_xy[0] - crop_size / 2)
        context_xmax = int(center_xy[0] + crop_size / 2)
        context_ymin = int(center_xy[1] - crop_size / 2)
        context_ymax = int(center_xy[1] + crop_size / 2)

        left_pad = max(0, -context_xmin)
        top_pad = max(0, -context_ymin)
        right_pad = max(0, context_xmax - W)
        bottom_pad = max(0, context_ymax - H)

        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad

        avg_channel = avg_channel[:, None, None]
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            new_img = img.new_zeros(N, C, H + top_pad + bottom_pad,
                                    W + left_pad + right_pad)
            new_img[..., top_pad:top_pad + H, left_pad:left_pad + W] = img
            if top_pad:
                new_img[..., :top_pad, left_pad:left_pad + W] = avg_channel
            if bottom_pad:
                new_img[..., H + top_pad:, left_pad:left_pad + W] = avg_channel
            if left_pad:
                new_img[..., :left_pad] = avg_channel
            if right_pad:
                new_img[..., W + left_pad:] = avg_channel
            crop_img = new_img[..., context_ymin:context_ymax + 1,
                               context_xmin:context_xmax + 1]
        else:
            crop_img = img[..., context_ymin:context_ymax + 1,
                           context_xmin:context_xmax + 1]

        crop_img = torch.nn.functional.interpolate(
            crop_img,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False)
        return crop_img

    def _bbox_clip(self, bbox, img_h, img_w):
        """Clip the bbox with [cx, cy, w, h] format."""
        bbox[0] = bbox[0].clamp(0., img_w)
        bbox[1] = bbox[1].clamp(0., img_h)
        bbox[2] = bbox[2].clamp(10., img_w)
        bbox[3] = bbox[3].clamp(10., img_h)
        return bbox

    def init(self, img, bbox):
        """Initialize the single object tracker in the first frame.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            bbox (Tensor): The given instance bbox of first frame that need be
                tracked in the following frames. The shape of the box is (4, )
                with [cx, cy, w, h] format.

        Returns:
            tuple(z_feat, avg_channel): z_feat is a tuple[Tensor] that
            contains the multi level feature maps of exemplar image,
            avg_channel is Tensor with shape (3, ), and denotes the padding
            values.
        """
        z_width = bbox[2] + self.test_cfg.context_amount * (bbox[2] + bbox[3])
        z_height = bbox[3] + self.test_cfg.context_amount * (bbox[2] + bbox[3])
        z_size = torch.round(torch.sqrt(z_width * z_height))
        avg_channel = torch.mean(img, dim=(0, 2, 3))
        z_crop = self.get_cropped_img(img, bbox[0:2],
                                      self.test_cfg.exemplar_size, z_size,
                                      avg_channel)
        z_feat = self.forward_template(z_crop)
        return z_feat, avg_channel

    def track(self, img, bbox, z_feat, avg_channel):
        """Track the box `bbox` of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            bbox (Tensor): The bbox in previous frame. The shape of the box is
                (4, ) in [cx, cy, w, h] format.
            z_feat (tuple[Tensor]): The multi level feature maps of exemplar
                image in the first frame.
            avg_channel (Tensor): of shape (3, ) denoting the padding values.

        Returns:
            tuple(best_score, best_bbox): best_score is a Tensor denoting the
            score of best_bbox, best_bbox is a Tensor of shape (4, ) in
            [cx, cy, w, h] format, and denotes the best tracked bbox in
            current frame.
        """
        z_width = bbox[2] + self.test_cfg.context_amount * (bbox[2] + bbox[3])
        z_height = bbox[3] + self.test_cfg.context_amount * (bbox[2] + bbox[3])
        z_size = torch.sqrt(z_width * z_height)

        x_size = torch.round(
            z_size * (self.test_cfg.search_size / self.test_cfg.exemplar_size))
        x_crop = self.get_cropped_img(img, bbox[0:2],
                                      self.test_cfg.search_size, x_size,
                                      avg_channel)

        x_feat = self.forward_search(x_crop)
        cls_score, bbox_pred = self.head(z_feat, x_feat)
        scale_factor = self.test_cfg.exemplar_size / z_size
        best_score, best_bbox = self.head.get_bbox(cls_score, bbox_pred, bbox,
                                                   scale_factor)

        # clip boundary
        best_bbox = self._bbox_clip(best_bbox, img.shape[2], img.shape[3])
        return best_score, best_bbox

    def simple_test(self, img, img_metas, gt_bboxes, **kwargs):
        """Test without augmentation.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.
            gt_bboxes (list[Tensor]): list of ground truth bboxes for each
                image with shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.

        Returns:
            dict[str : ndarray]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        assert len(img) == 1, 'only support batch_size=1 when testing'

        if frame_id == 0:
            gt_bboxes = gt_bboxes[0][0]
            self.memo = Dict()
            self.memo.bbox = bbox_xyxy_to_cxcywh(gt_bboxes)
            self.memo.z_feat, self.memo.avg_channel = self.init(
                img, self.memo.bbox)
            best_score = None
        else:
            best_score, self.memo.bbox = self.track(img, self.memo.bbox,
                                                    self.memo.z_feat,
                                                    self.memo.avg_channel)
            self.memo.bbox = torch.round(self.memo.bbox)

        bbox_pred = bbox_cxcywh_to_xyxy(self.memo.bbox)
        results = dict()
        if best_score is None:
            results['score'] = best_score
        else:
            results['score'] = best_score.cpu().numpy()
        results['bbox'] = bbox_pred.cpu().numpy()
        return results

    def forward_train(self, img, img_metas, gt_bboxes, search_img,
                      search_img_metas, search_gt_bboxes, is_positive_pairs,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input exemplar images.
                Typically H and W equal to 127.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each exemplar
                image with shape (1, 4) in [tl_x, tl_y, br_x, br_y] format.

            search_img (Tensor): of shape (N, 1, C, H, W) encoding input search
                images. 1 denotes there is only one search image for each
                exemplar image. Typically H and W equal to 255.

            search_img_metas (list[list[dict]]): The second list only has one
                element. The first list contains search image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            search_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                search image with shape (1, 5) in [0.0, tl_x, tl_y, br_x, br_y]
                format.

            is_positive_pairs (list[bool]): list of bool denoting whether each
                exemplar image and corresponding seach image is positive pair.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        search_img = search_img[:, 0]

        z_feat = self.forward_template(img)
        x_feat = self.forward_search(search_img)
        cls_score, bbox_pred = self.head(z_feat, x_feat)

        losses = dict()
        bbox_targets = self.head.get_targets(search_gt_bboxes,
                                             cls_score.shape[2:],
                                             is_positive_pairs)
        head_losses = self.head.loss(cls_score, bbox_pred, *bbox_targets)
        losses.update(head_losses)

        return losses
