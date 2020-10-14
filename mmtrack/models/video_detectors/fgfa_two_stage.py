import torch
from mmcv.cnn.bricks import ConvModule
from mmdet.models import build_detector
from mmdet.models.detectors import BaseDetector

from mmtrack.core import flow_warp_feats
from ..builder import MODELS, build_motion


@MODELS.register_module()
class FgfaTwoStage(BaseDetector):

    def __init__(self,
                 detector,
                 motion,
                 embed_network=dict(
                     in_channels=256,
                     out_channels=256,
                     kernel_size=3,
                     padding=1,
                     act_cfg=None),
                 train_cfg=None,
                 test_cfg=None):
        super(FgfaTwoStage, self).__init__()
        self.detector = build_detector(detector, train_cfg, test_cfg)
        self.motion = build_motion(motion)
        self.motion.init_weights()
        self.embed_network = ConvModule(**embed_network)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        return self.detector.extract_feat(img)

    def compute_adaptive_weights(self, x_single, target_index):
        x_single_embed = self.embed_network(x_single)
        x_single_embed = x_single_embed / x_single_embed.norm(
            p=2, dim=1, keepdim=True)
        x_target_embed = x_single_embed[[target_index]]
        ada_weights = torch.sum(
            x_single_embed * x_target_embed, dim=1, keepdim=True)
        ada_weights = ada_weights.softmax(dim=0)
        return ada_weights

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
            'fgfa video detectors only support 1 batch size per gpu for now.'
        is_video_data = img_metas[0]['is_video_data']
        if is_video_data:
            flow_imgs = torch.cat((img, ref_img[:, 0]), dim=1)
            for i in range(1, ref_img.shape[1]):
                flow_img = torch.cat((img, ref_img[:, i]), dim=1)
                flow_imgs = torch.cat((flow_imgs, flow_img), dim=0)
            flows = self.motion(flow_imgs, img_metas)

            all_imgs = torch.cat((img, ref_img[0]), dim=0)
            all_x = self.extract_feat(all_imgs)
            x = []
            for i in range(len(all_x)):
                ref_x_single = flow_warp_feats(all_x[i][1:], flows)
                x_single = torch.cat((all_x[i][[0]], ref_x_single), dim=0)
                ada_weights = self.compute_adaptive_weights(
                    x_single, target_index=0)
                x_single = torch.sum(
                    x_single * ada_weights, dim=0, keepdim=True)
                x.append(x_single)
        else:
            x = self.extract_feat(img)

        losses = dict()

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
        else:
            proposal_list = proposals

        roi_losses = self.detector.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    ref_img=None,
                    ref_img_metas=None,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        frame_id = img_metas[0]['frame_id']
        num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', -1)
        frame_stride = img_metas[0].get('frame_stride', -1)

        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                self.ref_imgs = ref_img[0]
                ref_x = self.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.ref_x = []
                for i in range(len(ref_x)):
                    self.ref_x.append(ref_x[i])
            x_frame = self.extract_feat(img)
        # test with fixed stride
        else:
            if frame_id == 0:
                self.ref_imgs = ref_img[0]
                ref_x = self.extract_feat(ref_img[0])
                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.ref_x = []
                # the features of img is same as ref_x[i][[num_left_ref_imgs]]
                x_frame = []
                for i in range(len(ref_x)):
                    self.ref_x.append(ref_x[i])
                    x_frame.append(ref_x[i][[num_left_ref_imgs]])
            elif frame_id % frame_stride == 0:
                assert ref_img is not None
                x_frame = []
                ref_x = self.extract_feat(ref_img[0])
                for i in range(len(ref_x)):
                    self.ref_x[i] = torch.cat((self.ref_x[i], ref_x[i]),
                                              dim=0)[1:]
                    x_frame.append(self.ref_x[i][[num_left_ref_imgs]])
                self.ref_imgs = torch.cat((self.ref_imgs, img), dim=0)[1:]
            else:
                assert ref_img is None
                x_frame = self.extract_feat(img)

        flow_imgs = torch.cat(
            (img.repeat(self.ref_imgs.shape[0], 1, 1, 1), self.ref_imgs),
            dim=1)
        flows = self.motion(flow_imgs, img_metas)

        x = []
        for i in range(len(x_frame)):
            x_single = flow_warp_feats(self.ref_x[i], flows)
            if frame_stride < 1:
                x_single = torch.cat((x_frame[i], x_single), dim=0)
                ada_weights = self.compute_adaptive_weights(
                    x_single, target_index=0)
            else:
                x_single[num_left_ref_imgs] = x_frame[i]
                ada_weights = self.compute_adaptive_weights(
                    x_single, target_index=num_left_ref_imgs)
            x_single = torch.sum(x_single * ada_weights, dim=0, keepdim=True)
            x.append(x_single)

        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
        else:
            proposal_list = proposals

        outs = self.detector.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        results = dict()
        results['bbox_results'] = outs[0]
        if len(outs) == 2:
            results['segm_results'] = outs[1]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
