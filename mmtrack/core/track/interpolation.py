import numpy as np


def tracklet_interpolation(results, min_frames=5, max_frames=20):
    """Interpolate tracklet.

    This function is proposed in
    "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
    `ByteTrack<https://arxiv.org/abs/2110.06864>`_.

    Args:
        results (ndarray): With shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score)
        min_frames (int, optional): The minimum frames of a tracklet that will
            be interpolated. Defaults to 5.
        max_frames (int, optional): The maximum frames of a tracklet that will
            be interpolated. Defaults to 20.

    Returns:
        ndarray: The interpolated results with shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score)
    """
    interpolation_results = []
    max_track_id = int(np.max(results[:, 1]))
    min_track_id = int(np.min(results[:, 1]))

    # perform interpolation for each tracklet
    for track_id in range(min_track_id, max_track_id + 1):
        inds = results[:, 1] == track_id
        tracklet = results[inds]
        num_frames = len(tracklet)
        if num_frames <= 2:
            continue

        interpolation_tracklet = tracklet
        if num_frames > min_frames:
            frame_ids = tracklet[:, 0]
            interpolation_results_per_tracklet = np.zeros((0, 7))
            # perform interpolation for each disconnected snippet in the
            # tracklet
            for i in np.where(np.diff(frame_ids) > 1)[0]:
                left_frame_id = frame_ids[i]
                right_frame_id = frame_ids[i + 1]
                num_interpolation_frames = int(right_frame_id - left_frame_id)

                if 1 < num_interpolation_frames < max_frames:
                    left_bbox = tracklet[i, 2:6]
                    right_bbox = tracklet[i + 1, 2:6]

                    # perform interpolation within the disconnected snippet
                    for j in range(1, num_interpolation_frames):
                        cur_bbox = j / (num_interpolation_frames) * (
                            right_bbox - left_bbox) + left_bbox
                        cur_result = np.ones((7, ))
                        cur_result[0] = j + left_frame_id
                        cur_result[1] = track_id
                        cur_result[2:6] = cur_bbox

                        interpolation_results_per_tracklet = np.concatenate(
                            (interpolation_results_per_tracklet,
                             cur_result[None]),
                            axis=0)

            interpolation_tracklet = np.concatenate(
                (interpolation_tracklet, interpolation_results_per_tracklet),
                axis=0)
        interpolation_results.append(interpolation_tracklet)
    interpolation_results = np.concatenate(interpolation_results)
    return interpolation_results[interpolation_results[:, 0].argsort()]
