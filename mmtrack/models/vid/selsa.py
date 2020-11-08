import torch
from mmdet.models import build_detector

from ..builder import MODELS
from .base import BaseVideoDetector


@MODELS.register_module()
class SELSA(BaseVideoDetector):

    def __init__(self,
                 detector,
                 pretrains=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SELSA, self).__init__()
        self.detector = build_detector(detector, train_cfg, test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrains)
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self, pretrain):
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if self.with_detector and pretrain.get('detector', False):
            self.init_module('detector', pretrain['detector'])
        if self.with_motion:
            self.init_module('motion', pretrain.get('motion', None))

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_instance_ids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ref_gt_instance_ids=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_proposals=None,
                      **kwargs):
        assert len(img) == 1, \
            'selsa video detector only supports 1 batch size per gpu for now.'

        all_imgs = torch.cat((img, ref_img[0]), dim=0)
        all_x = self.detector.extract_feat(all_imgs)
        x = []
        ref_x = []
        for i in range(len(all_x)):
            x.append(all_x[i][[0]])
            ref_x.append(all_x[i][1:])

        losses = dict()

        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas[0])
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        roi_losses = self.detector.roi_head.forward_train(
            x, ref_x, img_metas, proposal_list, ref_proposals_list, gt_bboxes,
            gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    ref_img=None,
                    ref_img_metas=None,
                    proposals=None,
                    ref_proposals=None,
                    rescale=False):
        """Test without augmentation."""
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', -1)
        frame_stride = img_metas[0].get('frame_stride', -1)

        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                self.buffer_img_metas = ref_img_metas[0]
                buffer_x = self.detector.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.buffer_x = []
                for i in range(len(buffer_x)):
                    self.buffer_x.append(buffer_x[i])
            x_current_frame = self.detector.extract_feat(img)

            x = x_current_frame
            ref_x = self.buffer_x
            ref_img_metas = self.buffer_img_metas
        # test with fixed stride
        else:
            if frame_id == 0:
                self.buffer_img_metas = ref_img_metas[0]
                buffer_x = self.detector.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.buffer_x = []
                # the features of img is same as ref_x[i][[num_left_ref_imgs]]
                x_current_frame = []
                for i in range(len(buffer_x)):
                    self.buffer_x.append(buffer_x[i])
                    x_current_frame.append(buffer_x[i][[num_left_ref_imgs]])
            elif frame_id % frame_stride == 0:
                assert ref_img is not None
                x_current_frame = []
                buffer_x = self.detector.extract_feat(ref_img[0])
                for i in range(len(buffer_x)):
                    self.buffer_x[i] = torch.cat(
                        (self.buffer_x[i], buffer_x[i]), dim=0)[1:]
                    x_current_frame.append(
                        self.buffer_x[i][[num_left_ref_imgs]])
                self.buffer_img_metas.extend(ref_img_metas[0])
                self.buffer_img_metas = self.buffer_img_metas[1:]
            else:
                assert ref_img is None
                x_current_frame = self.detector.extract_feat(img)

            x = x_current_frame
            ref_x = []
            for i in range(len(self.buffer_x)):
                ref_x_single = torch.cat(
                    (self.buffer_x[i][:num_left_ref_imgs],
                     self.buffer_x[i][num_left_ref_imgs + 1:]),
                    dim=0)
                ref_x.append(ref_x_single)
            ref_img_metas = []
            for i in range(len(self.buffer_img_metas)):
                if i != num_left_ref_imgs:
                    ref_img_metas.append(self.buffer_img_metas[i])

        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas)
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        outs = self.detector.roi_head.simple_test(
            x,
            ref_x,
            proposal_list,
            ref_proposals_list,
            img_metas,
            rescale=rescale)

        results = dict()
        results['bbox_results'] = outs[0]
        if len(outs) == 2:
            results['segm_results'] = outs[1]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
