import cv2
import numpy as np
import torch

from ..builder import MOTION


@MOTION.register_module()
class CameraMotionCompensation(object):

    def __init__(self,
                 warp_mode='cv2.MOTION_EUCLIDEAN',
                 num_iters=50,
                 stop_eps=0.001):
        self.warp_mode = eval(warp_mode)
        self.num_iters = num_iters
        self.stop_eps = stop_eps

    def get_warp_matrix(self, img, ref_img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.cpu().numpy()

        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ref_img = np.transpose(ref_img, (1, 2, 0))
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
        tl, br = bboxes[:, :2], bboxes[:, 2:]
        tl = torch.cat((tl, torch.ones(tl.shape[0], 1)), dim=1)
        br = torch.cat((br, torch.ones(tl.shape[0], 1)), dim=1)
        trans_tl = torch.mm(warp_matrix, tl.t()).t()
        trans_br = torch.mm(warp_matrix, br.t()).t()
        trans_bboxes = torch.cat((trans_tl, trans_br), dim=1)
        return trans_bboxes.to(bboxes.device)
