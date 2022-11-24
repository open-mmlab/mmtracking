# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmdet.models.builder import build_backbone, build_head
from torchvision.transforms.functional import normalize

from ..builder import MODELS
from .stark import Stark


@MODELS.register_module()
class MixFormer(Stark):
    """MixFormer: End-to-End Tracking with Iterative Mixed Attention.

    This single object tracker is the implementation of
    `MixFormer<https://arxiv.org/abs/2203.11082>`_.

    """

    def __init__(self,
                 backbone,
                 head=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Stark, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.update_interval = self.test_cfg['update_interval'][0]
        self.online_size = self.test_cfg['online_size'][0]
        self.max_score_decay = self.test_cfg['max_score_decay'][0]

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def set_online(self, template, online_template):
        self.backbone.set_online(template, online_template)

    def init(self, img, bbox):
        """Initialize the single object tracker in the first frame.

        Args:
            img: (Tensor): input image of shape (1, C, H, W).
            bbox (list | Tensor): in [cx, cy, w, h] format.
        """
        self.z_dict_list = []  # store templates
        # get the 1st template
        z_patch, _, z_mask = self.get_cropped_img(
            img, bbox, self.test_cfg['template_factor'],
            self.test_cfg['template_size']
        )  # z_patch pf shape [1,C,H,W]; z_mask of shape [1,H,W]
        z_patch = normalize(
            z_patch.squeeze() / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]).unsqueeze(0)
        with torch.no_grad():
            self.set_online(z_patch, z_patch)

        self.template = z_patch
        self.online_template = z_patch
        self.best_online_template = z_patch
        self.best_conf_score = -1.0
        self.online_forget_id = 0

    def update_template(self, img, bbox, conf_score):
        """Update the dynamic templates.

        Args:
            img (Tensor): of shape (1, C, H, W).
            bbox (list | ndarray): in [cx, cy, w, h] format.
            conf_score (float): the confidence score of the predicted bbox.
        """
        if conf_score > 0.5 and conf_score > self.best_conf_score:
            z_patch, _, z_mask = self.get_cropped_img(
                img,
                bbox,
                self.test_cfg['template_factor'],
                output_size=self.test_cfg['template_size'],
            )
            z_patch = normalize(
                z_patch.squeeze() / 255.,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ).unsqueeze(0)
            self.best_online_template = z_patch
            self.best_conf_score = conf_score
        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.best_online_template
            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat(
                    [self.online_template, self.best_online_template])
            else:
                self.online_template[self.
                                     online_forget_id:self.online_forget_id +
                                     1] = self.best_online_template
                self.online_forget_id = (self.online_forget_id +
                                         1) % self.online_size

            with torch.no_grad():
                self.set_online(self.template, self.online_template)

            self.best_conf_score = -1
            self.best_online_template = self.template

    def track(self, img, bbox):
        """Track the box `bbox` of previous frame to current frame `img`

        Args:
            img (Tensor): of shape (1, C, H, W).
            bbox (list | Tensor): The bbox in previous frame. The shape of the
                bbox is (4, ) in [x, y, w, h] format.
        """
        H, W = img.shape[2:]
        # get the t-th search region
        x_patch, resize_factor, x_mask = self.get_cropped_img(
            img, bbox, self.test_cfg['search_factor'],
            self.test_cfg['search_size']
        )  # bbox: of shape (x1,y1, w, h), x_mask: of shape (1, h, w)
        x_patch = normalize(
            x_patch.squeeze() / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]).unsqueeze(0)

        with torch.no_grad():
            x_patch.squeeze(1)
            template, search = self.backbone.forward_test(x_patch)
            out_dict = self.head(template, search)

        pred_box = out_dict['pred_bboxes']  # xyxy
        pred_box = self.mapping_bbox_back(pred_box, self.memo.bbox,
                                          resize_factor)
        pred_box = self._bbox_clip(pred_box, H, W, margin=10)

        # update template
        self.best_conf_score = self.best_conf_score * self.max_score_decay
        conf_score = -1.
        if self.head.score_decoder_head is not None:
            # get confidence score (whether the search region is reliable)
            conf_score = out_dict['pred_scores'].view(1).sigmoid().item()
            crop_bbox = bbox_xyxy_to_cxcywh(pred_box)
            self.update_template(img, crop_bbox, conf_score)

        return conf_score, pred_box

    def forward_train(self, imgs, img_metas, search_img, search_img_metas,
                      **kwargs):
        """forward of training.

        Args:
            img (Tensor): template images of shape (N, num_templates, C, H, W)
                Typically, there are 2 template images,
                and H and W are both equal to 128.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'image_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            search_img (Tensor): of shape (N, 1, C, H, W) encoding input search
                images. 1 denotes there is only one search image for each
                exemplar image. Typically H and W are both equal to 320.

            search_img_metas (list[list[dict]]): The second list only has one
                element. The first list contains search image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape' and
                'img_norm_cfg'. For details on the values of there keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for template
                images with shape (N, 4) in [tl_x, tl_y, br_x, br_y] format.

            padding_mask (Tensor): padding mask of template images.
                It's of shape (N, num_templates, H, W).
                Typically, there are 2 padding masks of tehmplate images, and
                H and W are both equal to that of template images.

            search_gt_bboxes (list[Tensor]): Ground truth bboxes for search
                images with shape (N, 5) in
                [0., tl_x, tl_y, br_x, br_y] format.

            search_padding_mask (Tensor): padding mask of search images.
                Its of shape (N, 1, H, W).
                There are 1 padding masks of search image, and
                H and W are both equal to that of search image.

            search_gt_labels (list[Tensor], optional): Ground truth labels for
                search images with shape (N, 2).

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        template, online_template = imgs[:, 0, ...], imgs[:, 1, ...]
        search = search_img.squeeze(1)
        template, search = self.backbone(template, online_template, search)

        # box head
        out_dict = self.head(template, search, **kwargs)

        # compute loss
        return out_dict
