# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch

from ..builder import MOTION


@MOTION.register_module()
class CameraMotionCompensation(object):
    """Camera motion compensation.

    Args:
        warp_mode (str): Warp mode in opencv.
        num_iters (int): Number of the iterations.
        stop_eps (float): Terminate threshold.
    """

    def __init__(self,
                 warp_mode='cv2.MOTION_EUCLIDEAN',
                 num_iters=50,
                 stop_eps=0.001):
        self.warp_mode = eval(warp_mode)
        self.num_iters = num_iters
        self.stop_eps = stop_eps

    def get_warp_matrix(self, img, ref_img):
        """Calculate warping matrix between two images."""
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.num_iters, self.stop_eps)
        cc, warp_matrix = cv2.findTransformECC(img, ref_img, warp_matrix,
                                               self.warp_mode, criteria, None,
                                               1)
        warp_matrix = torch.from_numpy(warp_matrix)
        return warp_matrix

    def warp_bboxes(self, bboxes, warp_matrix):
        """Warp bounding boxes according to the warping matrix."""
        tl, br = bboxes[:, :2], bboxes[:, 2:]
        tl = torch.cat((tl, torch.ones(tl.shape[0], 1).to(bboxes.device)),
                       dim=1)
        br = torch.cat((br, torch.ones(tl.shape[0], 1).to(bboxes.device)),
                       dim=1)
        trans_tl = torch.mm(warp_matrix, tl.t()).t()
        trans_br = torch.mm(warp_matrix, br.t()).t()
        trans_bboxes = torch.cat((trans_tl, trans_br), dim=1)
        return trans_bboxes.to(bboxes.device)

    def track(self, img, ref_img, tracks, num_samples, frame_id):
        """Tracking forward."""
        img = img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        ref_img = ref_img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        warp_matrix = self.get_warp_matrix(img, ref_img)

        bboxes = []
        num_bboxes = []
        for k, v in tracks.items():
            if int(v['frame_ids'][-1]) < frame_id - 1:
                _num = 1
            else:
                _num = min(num_samples, len(v.bboxes))
            num_bboxes.append(_num)
            bboxes.extend(v.bboxes[-_num:])
        bboxes = torch.cat(bboxes, dim=0)
        warped_bboxes = self.warp_bboxes(bboxes, warp_matrix.to(bboxes.device))

        warped_bboxes = torch.split(warped_bboxes, num_bboxes)
        for b, (k, v) in zip(warped_bboxes, tracks.items()):
            _num = b.shape[0]
            b = torch.split(b, [1] * _num)
            tracks[k].bboxes[-_num:] = b
        return tracks
