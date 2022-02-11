import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

from .. import builder
from ..registry import DETECTORS
from ..utils import ConvModule
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class FasterRCNNFGFATRoiAlignFlow512(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 flow_net=None,
                 embed_network=None,
                 use_troi=True,
                 sampling_point=4,
                 backbone_imgconv=-1,
                 use_TopKCos_weight_feats=False,
                 use_roi_adaptive_weights=False,
                 multi_head_att=1):
        super(FasterRCNNFGFATRoiAlignFlow512, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.flow_net = builder.build_backbone(flow_net)
        self.embed_network = ConvModule(
            embed_network['in_channel'],
            embed_network['out_channel'],
            3,
            padding=1,
            activation=None,
            inplace=False)
        self.use_troi = use_troi
        self.sampling_point = sampling_point
        self.iter_tmp = 0
        self.relu = torch.nn.ReLU(inplace=True)
        self.use_TopKCos_weight_feats = use_TopKCos_weight_feats
        self.use_roi_adaptive_weights = use_roi_adaptive_weights
        self.multi_head_att = multi_head_att
        if self.use_roi_adaptive_weights:
            assert backbone_imgconv > 0
            self.roi_embed_network = ConvModule(
                backbone_imgconv,
                backbone_imgconv,
                3,
                padding=1,
                activation=None,
                inplace=False)
        if backbone_imgconv > 0:
            self.with_backbone_imgconv = True
            self.backbone_imgconv = ConvModule(
                self.backbone.inplanes,
                backbone_imgconv,
                1,
                padding=0,
                activation=None,
                inplace=False)

    def flow_warp_feats(self, flow, x_ref_single):
        scale_factor = float(x_ref_single.shape[-1]) / flow.shape[-1]
        flow = torch.nn.functional.interpolate(
            flow, scale_factor=scale_factor, mode='bilinear')
        flow = flow * scale_factor

        H, W = x_ref_single.shape[-2:]
        h_grid, w_grid = torch.meshgrid(torch.arange(H), torch.arange(W))
        h_grid = h_grid.float().cuda().unsqueeze(0)
        w_grid = w_grid.float().cuda().unsqueeze(0)
        grid = torch.cat((w_grid, h_grid), dim=0).unsqueeze(0)
        grid = grid + flow
        grid[:, 0] = grid[:, 0] / W * 2 - 1
        grid[:, 1] = grid[:, 1] / H * 2 - 1
        grid = grid.permute(0, 2, 3, 1)

        x_single = torch.nn.functional.grid_sample(
            x_ref_single, grid, padding_mode='border')
        return x_single

    def compute_img_adaptive_weights(self, x_ref_single, ref_id):
        x_ref_embed = self.embed_network(x_ref_single)
        x_ref_embed = x_ref_embed / x_ref_embed.norm(p=2, dim=1, keepdim=True)
        x_cur_embed = x_ref_embed[[ref_id]]
        ada_weights = torch.sum(x_ref_embed * x_cur_embed, dim=1, keepdim=True)
        ada_weights = ada_weights.softmax(dim=0)
        return ada_weights

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        elif self.with_backbone_imgconv:
            x = self.backbone_imgconv(x[0])
            x = [self.relu(x)]
        return x

    def compute_adaptive_weights(self, x, ref_id=0):
        img_n, roi_n, roi_c, roi_h, roi_w = x.size()
        x = x.view(-1, roi_c, roi_h, roi_w)
        x_embed = self.roi_embed_network(x)
        # x_embed = x
        x_embed = x_embed.view(img_n * roi_n, self.multi_head_att, -1, roi_h,
                               roi_w)
        # x_embed = x_embed / x_embed.norm(p=2, dim=2, keepdim=True)
        x_embed = x_embed.view(img_n, roi_n, self.multi_head_att, -1, roi_h,
                               roi_w)
        x_cur_embed = x_embed[[ref_id]]
        ada_weights = torch.sum(
            x_embed * x_cur_embed, dim=3, keepdim=True) / (
                float(roi_c / self.multi_head_att)**0.5)
        # ada_weights = torch.sum(x_embed * x_cur_embed, dim=3, keepdim=True)
        ada_weights = ada_weights.expand(-1, -1, -1,
                                         int(roi_c / self.multi_head_att), -1,
                                         -1)
        ada_weights = ada_weights.contiguous().view(img_n, roi_n, roi_c, roi_h,
                                                    roi_w)
        ada_weights = ada_weights.softmax(dim=0)
        return ada_weights

    def temporal_roi_align(self, bbox_feats, x_ref, sampling_point=4):
        # bbox_embed = self.roi_embed(bbox_feats)
        # bbox_embed = bbox_feats[:, :64, :, :].clone()
        # img_embed = self.img_embed(x_ref)
        bbox_embed = bbox_feats
        img_embed = x_ref
        bbox_embed = bbox_embed / bbox_embed.norm(p=2, dim=1, keepdim=True)
        img_embed = img_embed / img_embed.norm(p=2, dim=1, keepdim=True)
        roi_n, c_embed, roi_h, roi_w = bbox_embed.size()
        img_n, c_embed, img_h, img_w = img_embed.size()
        bbox_embed = bbox_embed.permute(0, 2, 3, 1).contiguous()
        bbox_embed = bbox_embed.view(-1, c_embed)
        img_embed = img_embed.permute(1, 0, 2, 3).contiguous()
        img_embed = img_embed.view(c_embed, -1)
        cos_roi_img = bbox_embed.mm(img_embed)
        # cos_roi_img = torch.rand(12544, img_n * img_h * img_w).cuda()
        cos_roi_img = cos_roi_img.view(-1, img_n, img_h * img_w)
        values, indices = cos_roi_img.topk(
            k=sampling_point,
            dim=2,
            largest=True,
        )

        x_ref_reshape = x_ref.permute(2, 3, 0, 1).contiguous().view(
            -1, img_n, x_ref.shape[1])

        if self.use_TopKCos_weight_feats:
            values = values / values.norm(p=1, dim=2, keepdim=True)
        else:
            values[...] = 1.0 / float(sampling_point)
        t_bbox_feats = (x_ref_reshape[indices[:, 0], 0, :] *
                        values[:, 0].unsqueeze(-1)).sum(dim=1).unsqueeze(0)
        for i in range(1, img_n):
            t_bbox_feats = torch.cat([
                t_bbox_feats,
                (x_ref_reshape[indices[:, i], i, :] *
                 values[:, i].unsqueeze(-1)).sum(dim=1).unsqueeze(0)
            ],
                                     dim=0)
        t_bbox_feats = t_bbox_feats.view(img_n, roi_n, roi_h, roi_w,
                                         bbox_feats.shape[1]).permute(
                                             0, 1, 4, 2, 3)
        t_bbox_feats = torch.cat([bbox_feats.unsqueeze(0), t_bbox_feats],
                                 dim=0)
        if self.use_roi_adaptive_weights:
            adaptive_weights = self.compute_adaptive_weights(t_bbox_feats)
            t_bbox_feats = (t_bbox_feats * adaptive_weights).sum(dim=0)
        else:
            t_bbox_feats = t_bbox_feats.mean(dim=0)
        return t_bbox_feats

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      bef_img_meta=None,
                      bef_img=None,
                      aft_img_meta=None,
                      aft_img=None,
                      eq_flag=None):
        if self.iter_tmp == 0:
            eq_flag[...] = 0
            self.iter_tmp += 1
        if eq_flag[0]:
            x = self.extract_feat(img)
        else:
            img_flow = self.flow_net.prepare_img_for_flownet(img, img_meta)
            bef_img_flow = self.flow_net.prepare_img_for_flownet(
                bef_img, bef_img_meta)
            pair_bef_img = torch.cat((img_flow, bef_img_flow),
                                     dim=0).unsqueeze(dim=0)
            aft_img_flow = self.flow_net.prepare_img_for_flownet(
                aft_img, aft_img_meta)
            pair_aft_img = torch.cat((img_flow, aft_img_flow),
                                     dim=0).unsqueeze(dim=0)
            pair_img = torch.cat((pair_bef_img, pair_aft_img), dim=0)

            flow = self.flow_net(pair_img)

            backbone_img = torch.cat((img, bef_img, aft_img), dim=0)
            x_backbone_img = self.extract_feat(backbone_img)
            x = []
            if self.use_troi:
                x_ref = []
            for i in range(len(x_backbone_img)):
                x_single = self.flow_warp_feats(flow, x_backbone_img[i][1:])
                x_single = torch.cat((x_backbone_img[i][[0]], x_single), dim=0)
                ada_weights = self.compute_img_adaptive_weights(x_single, 0)
                x_single = torch.sum(
                    x_single * ada_weights, dim=0, keepdim=True)

                x.append(x_single)
                if self.use_troi:
                    x_ref.append(x_backbone_img[i][1:])

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            if eq_flag[0]:
                _ = 1
            else:
                if self.use_troi:
                    bbox_feats = self.temporal_roi_align(
                        bbox_feats,
                        x_ref[0],
                        sampling_point=self.sampling_point)

            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def forward_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        img_meta = img_meta[0]
        img = torch.cat(img, dim=0)
        img_flow = self.flow_net.prepare_img_for_flownet(img[[0]],
                                                         img_meta).unsqueeze(0)
        for i in range(1, img.shape[0]):
            img_flow_next = self.flow_net.prepare_img_for_flownet(
                img[[i]], img_meta).unsqueeze(0)
            img_flow = torch.cat((img_flow, img_flow_next), dim=0)
        x = self.extract_feat(img)

        if img.shape[0] > 1:
            self.cache_feats = [i for i in x]
            self.cache_img_flow = img_flow
            self.cur_img_id = int((img.shape[0] - 1) / 2)
        else:
            self.cache_img_flow = torch.cat((self.cache_img_flow, img_flow),
                                            dim=0)[1:]
            for i in range(len(x)):
                self.cache_feats[i] = torch.cat((self.cache_feats[i], x[i]),
                                                dim=0)[1:]

        img_flow_cur = self.cache_img_flow[[self.cur_img_id]]
        img_flow_cur = img_flow_cur.repeat(self.cache_img_flow.shape[0], 1, 1,
                                           1)
        pair_img_flow = torch.cat((img_flow_cur, self.cache_img_flow), dim=1)
        flow = self.flow_net(pair_img_flow)

        x = []
        x_ref = []
        for i in range(len(self.cache_feats)):
            x_single = self.flow_warp_feats(flow, self.cache_feats[i])
            x_single[self.cur_img_id] = self.cache_feats[i][self.cur_img_id]
            ada_weights = self.compute_img_adaptive_weights(
                x_single, self.cur_img_id)
            x_single = torch.sum(x_single * ada_weights, dim=0, keepdim=True)

            x_ref_single = torch.cat([
                self.cache_feats[i][0:self.cur_img_id],
                self.cache_feats[i][self.cur_img_id + 1:]
            ],
                                     dim=0)

            x.append(x_single)
            x_ref.append(x_ref_single)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x,
            img_meta,
            proposal_list,
            self.test_cfg.rcnn,
            rescale=rescale,
            x_ref=x_ref)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           x_ref=None):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)

        if self.use_troi:
            roi_feats = self.temporal_roi_align(
                roi_feats, x_ref[0], sampling_point=self.sampling_point)

        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels
