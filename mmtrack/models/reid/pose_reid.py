from typing import Optional, List
from mmtrack.registry import MODELS
from mmtrack.structures import ReIDDataSample
from mmengine.model import BaseModel
import torch
from mmengine.dataset import Compose
import numpy as np
from mmpose.datasets.transforms import LoadImage, GetBBoxCenterScale, PackPoseInputs


@MODELS.register_module()
class PoseReID(BaseModel):

    def __init__(self,
                 base_reid: Optional[dict] = None,
                 pose_model: Optional[dict] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.base_reid = MODELS.build(base_reid)
        self.pose_model = MODELS.build(pose_model)

        self.pose_pipeline = Compose(
            [LoadImage(),
             GetBBoxCenterScale(padding=1.0),
             PackPoseInputs()])

        self.pose_embbedder = FullBodyPoseEmbedder()

    @property
    def head(self):
        return self.base_reid.head

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[ReIDDataSample]] = None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            reid_results = self.base_reid(inputs, data_samples, mode)

            pose_data = []
            bboxes_ = []
            for input in inputs:
                img = input.detach().moveaxis(0, -1).cpu().numpy()
                height, width, _ = img.shape

                bboxes = np.array([[0, 0, width, height]], dtype=np.float32)
                bboxes_.append(bboxes)

                data = self.pose_pipeline(dict(img=img, bbox=bboxes))
                pds = data['data_samples']

                pds.gt_instances.bbox_scores = np.ones(1)
                pds.set_field((width, height),
                              'input_size',
                              field_type='metainfo')
                pds.set_field(
                    (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15),
                    'flip_indices',
                    field_type='metainfo')

                pose_data.append(pds)

            pose_results = self.pose_model.predict(inputs, pose_data)
        else:
            raise NotImplementedError(f'PoseReID does not support mode {mode}')

        bboxes_ = np.concatenate(bboxes_, axis=0)
        bboxes_ = torch.from_numpy(bboxes_).to(reid_results.device)

        pose_embedded = self.pose_embbedder(pose_results,
                                            bboxes_).to(reid_results.device)
        embedded = torch.cat((reid_results, pose_embedded), dim=1)
        return embedded


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
        pose_embeddings = []
        for k in range(len(pose_results)):
            w1, h1, w2, h2 = bboxes[k]

            landmarks = pose_results[k].pred_instances.keypoints.reshape(-1, 2)
            for i in range(landmarks.shape[0]):
                w, h = landmarks[i]
                landmarks[i][0] = (w - w1) / (w2 - w1)
                landmarks[i][1] = (h - h1) / (h2 - h1)
            pose_embeddings.append(self.embbed(landmarks))

        pose_embeddings = torch.from_numpy(np.stack(pose_embeddings, axis=0))

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
