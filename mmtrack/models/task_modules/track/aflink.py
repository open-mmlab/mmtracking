from typing import Optional

import torch
import numpy as np
from torch import nn
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from mmengine.model import BaseModule
from mmengine.runner.checkpoint import load_checkpoint

from mmtrack.registry import MODELS
INFINITY = 1e5


class TemporalBlock(BaseModule):
    def __init__(self, cin, cout):
        super(TemporalBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, (7, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bnf = nn.BatchNorm1d(cout)
        self.bnx = nn.BatchNorm1d(cout)
        self.bny = nn.BatchNorm1d(cout)

    def bn(self, x):
        x[:, :, :, 0] = self.bnf(x[:, :, :, 0])
        x[:, :, :, 1] = self.bnx(x[:, :, :, 1])
        x[:, :, :, 2] = self.bny(x[:, :, :, 2])
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FusionBlock(BaseModule):
    def __init__(self, cin, cout):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, (1, 3), bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Classifier(BaseModule):
    def __init__(self, cin):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(cin*2, cin//2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(cin//2, 2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@MODELS.register_module()
class AFLinkModel(BaseModule):
    """Appearance-Free Link Model."""

    def __init__(self):
        super(AFLinkModel, self).__init__()
        self.TemporalModule_1 = nn.Sequential(
            TemporalBlock(1, 32),
            TemporalBlock(32, 64),
            TemporalBlock(64, 128),
            TemporalBlock(128, 256)
        )
        self.TemporalModule_2 = nn.Sequential(
            TemporalBlock(1, 32),
            TemporalBlock(32, 64),
            TemporalBlock(64, 128),
            TemporalBlock(128, 256)
        )
        self.FusionBlock_1 = FusionBlock(256, 256)
        self.FusionBlock_2 = FusionBlock(256, 256)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(256)

    def forward(self, x1, x2):
        assert not self.training, "Only testing is supported for AFLink."
        x1 = x1[:, :, :, :3]
        x2 = x2[:, :, :, :3]
        x1 = self.TemporalModule_1(x1)  # [B,1,30,3] -> [B,256,6,3]
        x2 = self.TemporalModule_2(x2)
        x1 = self.FusionBlock_1(x1)
        x2 = self.FusionBlock_2(x2)
        x1 = self.pooling(x1).squeeze(-1).squeeze(-1)
        x2 = self.pooling(x2).squeeze(-1).squeeze(-1)
        y = self.classifier(x1, x2)
        y = torch.softmax(y, dim=1)[0, 1]
        return y


def data_transform(track1, track2, length=30):
    # fill or cut track1
    length_1 = track1.shape[0]
    track1 = track1[-length:] if length_1 >= length else \
        np.pad(track1, ((length - length_1, 0), (0, 0)))
    
    # fill or cut track1
    length_2 = track2.shape[0]
    track2 = track2[:length] if length_2 >= length else \
        np.pad(track2, ((0, length - length_2), (0, 0)))

    # min-max normalization
    min_ = np.concatenate((track1, track2), axis=0).min(axis=0)
    max_ = np.concatenate((track1, track2), axis=0).max(axis=0)
    subtractor = (max_ + min_) / 2
    divisor = (max_ - min_) / 2 + 1e-5
    track1 = (track1 - subtractor) / divisor
    track2 = (track2 - subtractor) / divisor

    # numpy to torch
    track1 = torch.tensor(track1, dtype=torch.float)
    track2 = torch.tensor(track2, dtype=torch.float)

    # unsqueeze channel=1
    track1 = track1.unsqueeze(0).unsqueeze(0).cuda()  # TODO: device
    track2 = track2.unsqueeze(0).unsqueeze(0).cuda()
    return track1, track2


def appearance_free_link(tracks: np.array,
                         temporal_threshold: tuple = (0, 30),
                         spatial_threshold: int = 75,
                         confidence_threshold: float = 0.95,
                         checkpoint: Optional[str] = None) -> np.array:
    """Appearance-Free Link method.

    This method is proposed in
    "StrongSORT: Make DeepSORT Great Again"
    `StrongSORT<https://arxiv.org/abs/2202.13514>`_.

    Args:
        tracks (ndarray): With shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score).
        temporal_threshold (tuple, optional): The temporal constraint
            for tracklets association. Defaults to (0, 30).
        spatial_threshold (int, optional): The spatial constraint for
            tracklets association. Defaults to 75.
        confidence_threshold (float, optional): The minimum confidence
            threshold for tracklets association. Defaults to 0.95.
        checkpoint (Optional[str], optional): Checkpoint path. Defaults to
            None.
    Returns:
        ndarray: The interpolated tracks with shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score)
    """
    model = AFLinkModel()
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
    model.cuda()
    model.eval()

    fn_l2 = lambda x, y: np.sqrt(x ** 2 + y ** 2)

    # sort tracks by the frame id
    tracks = tracks[np.argsort(tracks[:, 0])]

    # gather tracks information
    id2info = defaultdict(list)
    for row in tracks:
        frame_id, track_id, x1, y1, x2, y2 = row[:6]
        id2info[track_id].append([frame_id, x1, y1, x2 - x1, y2 - y1])
    id2info = {k: np.array(v) for k, v in id2info.items()}
    num_track = len(id2info)
    track_ids = np.array(list(id2info))
    cost_matrix = np.full((num_track, num_track), INFINITY)

    # compute the cost matrix
    for i, id_i in enumerate(track_ids):
        for j, id_j in enumerate(track_ids):
            if id_i == id_j:
                continue
            info_i, info_j = id2info[id_i], id2info[id_j]
            frame_i, box_i = info_i[-1][0], info_i[-1][1: 3]
            frame_j, box_j = info_j[0][0], info_j[0][1: 3]
            # temporal constraint
            if not temporal_threshold[0] <= \
                   frame_j - frame_i <= temporal_threshold[1]:
                continue
            # spatial constraint
            if fn_l2(box_i[0] - box_j[0], box_i[1] - box_j[1]) \
                    > spatial_threshold:
                continue
            # confidence constraint
            track_i, track_j = data_transform(info_i, info_j)
            confidence = model(track_i, track_j).detach().cpu().numpy()
            if confidence >= confidence_threshold:
                cost_matrix[i, j] = 1 - confidence

    # linear assignment
    indices = linear_sum_assignment(cost_matrix)
    _id2id = dict()  # the temporary assignment results
    id2id = dict()  # the final assignment results
    for i, j in zip(indices[0], indices[1]):
        if cost_matrix[i, j] < INFINITY:
            _id2id[i] = j
    for k, v in _id2id.items():
        if k in id2id:
            id2id[v] = id2id[k]
        else:
            id2id[v] = k

    # link
    for k, v in id2id.items():
        tracks[tracks[:, 1] == k, 1] = v

    # deduplicate
    _, index = np.unique(tracks[:, :2], return_index=True, axis=0)

    return tracks[index]
