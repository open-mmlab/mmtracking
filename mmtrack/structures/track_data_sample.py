# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement, InstanceData


class TrackDataSample(BaseDataElement):
    """A data structure interface of MMTracking. They are used as interfaces
    between different components.

    The attributes in ``TrackDataSample`` are divided into several parts:

        - ``gt_instances``(InstanceData): Ground truth of instance annotations
            in key frames.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing in key frames.
        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors in key frames.
        - ``ref_gt_instances``(InstanceData): Ground truth of instance
            annotations in reference frames.
        - ``ref_ignored_instances``(InstanceData): Instances to be ignored
            during training/testing in reference frames.
        - ``ref_proposals``(InstanceData): Region proposals used in two-stage
            detectors in reference frames.
        - ``pred_det_instances``(InstanceData): Detection instances of model
            predictions in key frames.
        - ``pred_track_instances``(InstanceData): Tracking instances of model
            predictions in key frames.
    """

    # Typically used in key frames.
    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    # Typically used in key frames.
    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, '_ignored_instances', dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    # Typically used in key frames.
    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    # Typically denotes the detection results of key frame
    @property
    def pred_det_instances(self) -> InstanceData:
        return self._pred_det_instances

    @pred_det_instances.setter
    def pred_det_instances(self, value: InstanceData):
        self.set_field(value, '_pred_det_instances', dtype=InstanceData)

    @pred_det_instances.deleter
    def pred_det_instances(self):
        del self._pred_det_instances

    # Typically denotes the tracking results of key frame
    @property
    def pred_track_instances(self) -> InstanceData:
        return self._pred_track_instances

    @pred_track_instances.setter
    def pred_track_instances(self, value: InstanceData):
        self.set_field(value, '_pred_track_instances', dtype=InstanceData)

    @pred_track_instances.deleter
    def pred_track_instances(self):
        del self._pred_track_instances
