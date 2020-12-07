from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class TracktorTracker(BaseTracker):

    def __init__(self,
                 obj_score_thr=0.3,
                 reg_obj_score_thr=0.3,
                 nms_thr=0.3,
                 reg_nms_thr=0.3,
                 reid_sim_thr=0.5,
                 reid_iou_thr=0.5,
                 **kwargs):
        super().__init__(**kwargs)

    def track(self, img, model, bboxes, labels, frame_id):
        if self.with_align:
            pass

        if model.with_motion:
            pass

        # regression

        if model.with_reid:
            pass
