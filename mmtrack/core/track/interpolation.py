# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def _interpolate_track(track, track_id, max_num_frames=20):
    """Interpolate a track linearly to make the track more complete.

    Args:
        track (ndarray): With shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score).
        max_num_frames (int, optional): The maximum disconnected length in the
            track. Defaults to 20.

    Returns:
        ndarray: The interpolated track with shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score)
    """
    assert (track[:, 1] == track_id).all(), \
        'The track id should not changed when interpolate a track.'

    frame_ids = track[:, 0]
    interpolated_track = np.zeros((0, 7))
    # perform interpolation for the disconnected frames in the track.
    for i in np.where(np.diff(frame_ids) > 1)[0]:
        left_frame_id = frame_ids[i]
        right_frame_id = frame_ids[i + 1]
        num_disconnected_frames = int(right_frame_id - left_frame_id)

        if 1 < num_disconnected_frames < max_num_frames:
            left_bbox = track[i, 2:6]
            right_bbox = track[i + 1, 2:6]

            # perform interpolation for two adjacent tracklets.
            for j in range(1, num_disconnected_frames):
                cur_bbox = j / (num_disconnected_frames) * (
                    right_bbox - left_bbox) + left_bbox
                cur_result = np.ones((7, ))
                cur_result[0] = j + left_frame_id
                cur_result[1] = track_id
                cur_result[2:6] = cur_bbox

                interpolated_track = np.concatenate(
                    (interpolated_track, cur_result[None]), axis=0)

    interpolated_track = np.concatenate((track, interpolated_track), axis=0)
    return interpolated_track


def interpolate_tracks(tracks, min_num_frames=5, max_num_frames=20):
    """Interpolate tracks linearly to make tracks more complete.

    This function is proposed in
    "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
    `ByteTrack<https://arxiv.org/abs/2110.06864>`_.

    Args:
        tracks (ndarray): With shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score).
        min_num_frames (int, optional): The minimum length of a track that will
            be interpolated. Defaults to 5.
        max_num_frames (int, optional): The maximum disconnected length in
            a track. Defaults to 20.

    Returns:
        ndarray: The interpolated tracks with shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score)
    """
    max_track_id = int(np.max(tracks[:, 1]))
    min_track_id = int(np.min(tracks[:, 1]))

    # perform interpolation for each track
    interpolated_tracks = []
    for track_id in range(min_track_id, max_track_id + 1):
        inds = tracks[:, 1] == track_id
        track = tracks[inds]
        num_frames = len(track)
        if num_frames <= 2:
            continue

        if num_frames > min_num_frames:
            interpolated_track = _interpolate_track(track, track_id,
                                                    max_num_frames)
        else:
            interpolated_track = track
        interpolated_tracks.append(interpolated_track)

    interpolated_tracks = np.concatenate(interpolated_tracks)
    return interpolated_tracks[interpolated_tracks[:, 0].argsort()]
