# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from motmetrics.lap import linear_sum_assignment
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.structures.bbox import bbox_xyxy_to_cxcyah
from mmtrack.utils import OptConfigType, imrenormalize
from .base_tracker import BaseTracker

from mmengine.dataset import Compose
import numpy as np
from mmpose.datasets.transforms import LoadImage, GetBBoxCenterScale, PackPoseInputs


@MODELS.register_module()
class MySORTTracker(BaseTracker):
    """Tracker for SORT/DeepSORT.

    Args:
        obj_score_thr (float, optional): Threshold to filter the objects.
            Defaults to 0.3.
        reid (dict, optional): Configuration for the ReID model.
            - num_samples (int, optional): Number of samples to calculate the
                feature embeddings of a track. Default to 10.
            - image_scale (tuple, optional): Input scale of the ReID model.
                Default to (256, 128).
            - img_norm_cfg (dict, optional): Configuration to normalize the
                input. Default to None.
            - match_score_thr (float, optional): Similarity threshold for the
                matching process. Default to 2.0.
        match_iou_thr (float, optional): Threshold of the IoU matching process.
            Defaults to 0.7.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
    """

    def __init__(self,
                 obj_score_thr: float = 0.3,
                 reid: dict = dict(
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0),
                 match_iou_thr: float = 0.7,
                 num_tentatives: int = 3,
                 biou: bool = False,
                 pose: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.obj_score_thr = obj_score_thr
        self.reid = reid
        self.match_iou_thr = match_iou_thr
        self.num_tentatives = num_tentatives
        self.biou = biou
        self.pose = pose

        self.pose_pipeline = Compose(
            [LoadImage(),
             GetBBoxCenterScale(padding=1.0),
             PackPoseInputs()])

        self.pose_embbedder = FullBodyPoseEmbedder()

    @property
    def confirmed_ids(self) -> List:
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    def init_track(self, id: int, obj: Tuple[Tensor]) -> None:
        """Initialize a track."""
        super().init_track(id, obj)
        self.tracks[id].tentative = True
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id: int, obj: Tuple[Tensor]) -> None:
        """Update a track."""
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def pop_invalid_tracks(self, frame_id: int) -> None:
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            # case1: disappeared frames >= self.num_frames_retrain
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            # case2: tentative tracks but not matched in this frame
            case2 = v.tentative and v['frame_ids'][-1] != frame_id
            if case1 or case2:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def track(self,
              model: torch.nn.Module,
              img: Tensor,
              feats: List[Tensor],
              data_sample: TrackDataSample,
              data_preprocessor: OptConfigType = None,
              rescale: bool = False,
              **kwargs) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                SORT method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.
            data_preprocessor (dict or ConfigDict, optional): The pre-process
               config of :class:`TrackDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_det_instances.bboxes
        labels = data_sample.pred_det_instances.labels
        scores = data_sample.pred_det_instances.scores

        frame_id = metainfo.get('frame_id', -1)
        if frame_id == 0:
            self.reset()
        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.with_reid:
            if self.reid.get('img_norm_cfg', False):
                img_norm_cfg = dict(
                    mean=data_preprocessor['mean'],
                    std=data_preprocessor['std'],
                    to_bgr=data_preprocessor['rgb_to_bgr'])
                reid_img = imrenormalize(img, img_norm_cfg,
                                         self.reid['img_norm_cfg'])
            else:
                reid_img = img.clone()

        valid_inds = scores > self.obj_score_thr
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]

        print()
        print('number of bboxes: ', bboxes.shape[0])
        print('reid_image:', reid_img.shape)

        if self.empty or bboxes.size(0) == 0:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += num_new_tracks
            if self.with_reid:
                crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale)
                if crops.size(0) > 0:
                    embeds = model.reid(crops, mode='tensor')
                    if self.pose:
                        pose_results, pose_embedded = self.get_pose_embedded(
                            bboxes.clone(), scores.clone(), metainfo, reid_img,
                            crops, model.pose)
                        embeds = torch.cat(
                            (embeds, pose_embedded.to(embeds.device)), dim=1)
                else:
                    pose_embedded = crops.new_zeros((0, 46))
                    embeds = crops.new_zeros((0, model.reid.head.out_channels))

        else:
            ids = torch.full((bboxes.size(0), ), -1,
                             dtype=torch.long).to(bboxes.device)

            # motion_bboxes = self.get('bboxes', self.ids)
            # print('motion_bboxes: ', len(motion_bboxes))
            # print(motion_bboxes[0])

            # print('number of tracks: ', len(self.tracks))
            # print(self.tracks[0].keys())
            # print(self.tracks[0]['ids'])
            # print(self.tracks[0]['bboxes'])
            # print(self.tracks[0]['frame_ids'])
            # print(self.tracks[0]['mean'].shape)
            # print(self.tracks[0]['covariance'].shape)

            # motion
            if model.with_motion:
                # print('motion')
                self.tracks, costs = model.motion.track(
                    self.tracks, bbox_xyxy_to_cxcyah(bboxes))

            # motion_bboxes = self.get('bboxes', self.ids)
            # print(motion_bboxes[0])

            active_ids = self.confirmed_ids
            if self.with_reid:
                crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale)

                embeds = model.reid(crops, mode='tensor')

                if self.pose:
                    pose_results, pose_embedded = self.get_pose_embedded(
                        bboxes.clone(), scores.clone(), metainfo, reid_img,
                        crops, model.pose)

                    embeds = torch.cat(
                        (embeds, pose_embedded.to(embeds.device)), dim=1)

                print('reid_mtaching')
                print('active_ids: ', len(active_ids))

                # reid
                if len(active_ids) > 0:
                    track_embeds = self.get(
                        'embeds',
                        active_ids,
                        self.reid.get('num_samples', None),
                        behavior='mean')

                    reid_dists = torch.cdist(track_embeds, embeds)

                    # support multi-class association
                    track_labels = torch.tensor([
                        self.tracks[id]['labels'][-1] for id in active_ids
                    ]).to(bboxes.device)
                    cate_match = labels[None, :] == track_labels[:, None]
                    cate_cost = (1 - cate_match.int()) * 1e6
                    reid_dists = (reid_dists + cate_cost).cpu().numpy()

                    valid_inds = [list(self.ids).index(_) for _ in active_ids]
                    reid_dists[~np.isfinite(costs[valid_inds, :])] = np.nan

                    row, col = linear_sum_assignment(reid_dists)
                    for r, c in zip(row, col):
                        dist = reid_dists[r, c]
                        if not np.isfinite(dist):
                            continue
                        if dist <= self.reid['match_score_thr']:
                            ids[c] = active_ids[r]

            active_ids = [
                id for id in self.ids if id not in ids
                and self.tracks[id].frame_ids[-1] == frame_id - 1
            ]

            print('biou_mtaching')
            print('active_ids: ', len(active_ids))
            if len(active_ids) > 0:
                active_dets = torch.nonzero(ids == -1).squeeze(1)
                track_bboxes = self.get('bboxes', active_ids)

                # print(active_ids)
                # print(track_bboxes[0])

                if self.biou:
                    ious = bbox_overlaps(
                        expanse_bboxes(track_bboxes),
                        expanse_bboxes(bboxes[active_dets]))

                else:
                    ious = bbox_overlaps(track_bboxes, bboxes[active_dets])

                # support multi-class association
                track_labels = torch.tensor([
                    self.tracks[id]['labels'][-1] for id in active_ids
                ]).to(bboxes.device)
                cate_match = labels[None, active_dets] == track_labels[:, None]
                cate_cost = (1 - cate_match.int()) * 1e6

                dists = (1 - ious + cate_cost).cpu().numpy()

                row, col = linear_sum_assignment(dists)
                for r, c in zip(row, col):
                    dist = dists[r, c]
                    if dist < 1 - self.match_iou_thr:
                        ids[active_dets[c]] = active_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            embeds=embeds if self.with_reid else None,
            frame_ids=frame_id)

        # update pred_track_instances
        pred_track_instances = InstanceData()
        pred_track_instances.bboxes = bboxes
        pred_track_instances.labels = labels
        pred_track_instances.scores = scores
        pred_track_instances.instances_id = ids
        if self.with_reid and self.pose:
            pred_track_instances.pose = pose_results

        return pred_track_instances

    def prepare_pose_data(self, img, bboxes, scores, crops):
        print('prepare_pose_data')
        pose_data = []

        for bbox, score, crop in zip(bboxes, scores, crops):
            data = self.pose_pipeline(dict(img=img,
                                           bbox=bbox[None]))  # shape (1, 4)
            pds = data['data_samples']
            pds.gt_instances.bbox_scores = score.reshape(1)
            pds.set_field(
                (crop.shape[2], crop.shape[1]),  # w, h
                'input_size',
                field_type='metainfo')
            pds.set_field(
                (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15),
                'flip_indices',
                field_type='metainfo')

            pose_data.append(pds)
        return pose_data

    def draw_img(self, bboxes, img, pose_results):
        print('draw_img')
        import cv2
        mean = np.array([[[123.675, 116.28, 103.53]]])
        std = np.array([[[58.395, 57.12, 57.375]]])
        img = img * std + mean

        cv2.imwrite('image.jpg', img[:, :, ::-1])
        img = cv2.imread('image.jpg')

        color = (255, 255, 0)
        thickness = 2

        for k in range(bboxes.shape[0]):
            start_point = (int(bboxes[k][0]), int(bboxes[k][1]))
            end_point = (int(bboxes[k][2]), int(bboxes[k][3]))
            img = cv2.rectangle(img, start_point, end_point, color, thickness)

            landmarks = pose_results[k].pred_instances.keypoints.reshape(-1, 2)
            for i in range(landmarks.shape[0]):
                center_coordinates = (int(landmarks[i][0]),
                                      int(landmarks[i][1]))
                radius = 3
                color = (100, 255, 100)
                thickness = 1
                img = cv2.circle(img, center_coordinates, radius, color,
                                 thickness)

        cv2.imwrite('image1.jpg', img)

    def get_pose_embedded(self, bboxes, scores, metainfo, img, crops,
                          pose_estimator):

        bboxes = bboxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        img = img.squeeze().detach().moveaxis(0, -1).cpu().numpy()

        factor_x, factor_y = metainfo['scale_factor']
        bboxes_scale = bboxes[:, :4] * np.array(
            [factor_x, factor_y, factor_x, factor_y])

        pose_data = self.prepare_pose_data(img, bboxes_scale, scores, crops)
        pose_results = pose_estimator.predict(crops, pose_data)
        self.draw_img(bboxes_scale, img, pose_results)

        pose_embedded = self.pose_embbedder(pose_results, bboxes_scale)

        for k in range(len(pose_results)):
            keypoints = pose_results[k].pred_instances.keypoints[0]
            keypoints /= np.array([factor_x, factor_y])
        return pose_results, pose_embedded


def expanse_bboxes(bboxes, b: float = 0.3):
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]

    w_center = (bboxes[:, 2] + bboxes[:, 0]) / 2
    h_center = (bboxes[:, 3] + bboxes[:, 1]) / 2

    w_expanse = (2 * b + 1) * w
    h_expanse = (2 * b + 1) * h

    return torch.stack((
        (w_center - w_expanse / 2),
        (h_center - h_expanse / 2),
        (w_center + w_expanse / 2),
        (h_center + h_expanse / 2),
    ),
                       dim=1)


class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye',
            'right_eye',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle',
        ]

    def embbed(self, landmarks):
        """
        Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(
            self._landmark_names), 'Unexpected number of landmarks: {}'.format(
                landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        embedding = (embedding + 1) / 2

        return embedding.reshape(-1)

    def __call__(self, pose_results, bboxes):
        print('pose embedded')
        pose_embeddings = []
        for k in range(len(pose_results)):
            w1, h1, w2, h2 = bboxes[k]

            landmarks = np.copy(
                pose_results[k].pred_instances.keypoints.reshape(-1, 2))

            for i in range(landmarks.shape[0]):
                w, h = landmarks[i]
                landmarks[i][0] = (w - w1) / (w2 - w1)
                landmarks[i][1] = (h - h1) / (h2 - h1)
            # print(landmarks)
            pose_embeddings.append(self.embbed(landmarks))

        pose_embeddings = torch.from_numpy(np.stack(pose_embeddings, axis=0))
        # print(pose_embeddings[0])
        return pose_embeddings

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        # landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index(
            'right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.
            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder',
                                           'right_shoulder')),
            self._get_distance_by_names(landmarks, 'left_shoulder',
                                        'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder',
                                        'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow',
                                        'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),
            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee',
                                        'right_ankle'),

            # Two joints.
            self._get_distance_by_names(landmarks, 'left_shoulder',
                                        'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder',
                                        'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.
            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.
            self._get_distance_by_names(landmarks, 'left_shoulder',
                                        'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder',
                                        'right_ankle'),
            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.
            self._get_distance_by_names(landmarks, 'left_elbow',
                                        'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),
            self._get_distance_by_names(landmarks, 'left_wrist',
                                        'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle',
                                        'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from
