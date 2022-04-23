# Copyright (c) OpenMMLab. All rights reserved.
# The codes are modified from https://github.com/votchallenge/toolkit/blob/master/vot/analysis/supervised.py # noqa: E501
import numpy as np

try:
    import vot
    from vot.analysis import is_special
    from vot.region import Polygon, Rectangle, Special
    from vot.region import calculate_overlaps as calculate_region_overlaps
except ImportError:
    vot = None


def bbox2region(bbox):
    """Convert bbox to Rectangle or Polygon Class object.

    Args:
        bbox (ndarray): the format of rectangle bbox is (x1, y1, w, h);
            the format of polygon is (x1, y1, x2, y2, ...).

    Returns:
        Rectangle or Polygon Class object.
    """
    if vot is None:
        raise ImportError(
            'Please run'
            ' pip install git+https://github.com/votchallenge/toolkit.git '
            'to manually install vot-toolkit')

    if len(bbox) == 1:
        return Special(bbox[0])
    elif len(bbox) == 4:
        return Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
    elif len(bbox) % 2 == 0 and len(bbox) > 4:
        return Polygon([(x_, y_) for x_, y_ in zip(bbox[::2], bbox[1::2])])
    else:
        raise NotImplementedError(
            f'The length of bbox is {len(bbox)}, which is not supported')


def trajectory2region(trajectory):
    """Convert bbox trajectory to Rectangle or Polygon Class object trajectory.

    Args:
        trajectory (list[ndarray]): The outer list contains bbox of
            each frame in a video. The bbox is a ndarray.

    Returns:
        List: contains the Region Class object of each frame in a
            trajectory.
    """
    traj_region = []
    for bbox in trajectory:
        traj_region.append(bbox2region(bbox))
    return traj_region


def locate_failures_inits(trajectory):
    """locate the failure frame and initialized frame in a trajectory.

    Args:
        trajectory (list[ndarray]): list of tracking results.

    Returns:
        fail_inds (list): index of failed frame in a trajectory.
        init_inds (list): index of initialized frame in a trajectory.
    """
    fail_inds = []
    init_inds = []
    for i, bbox in enumerate(trajectory):
        if len(bbox) == 1:
            if bbox[0] == 1.:
                init_inds.append(i)
            elif bbox[0] == 2.:
                fail_inds.append(i)
    return fail_inds, init_inds


def count_failures(trajectory):
    """count the number of failed frame in a trajectory.

    Args:
        trajectory (list[ndarray]): list of tracking results.

    Returns:
        List: the number of failed frame in a trajectory.
    """
    num_fails = 0
    for bbox in trajectory:
        if len(bbox) == 1 and bbox[0] == 2.:
            num_fails += 1
    return num_fails


def calc_accuracy(gt_trajectory,
                  pred_trajectory,
                  burnin=10,
                  ignore_unknown=True,
                  video_wh=None):
    """Calculate accuracy over the sequence.

    Args:
        gt_trajectory (list[list]): list of bboxes
        pred_trajectory (list[ndarray]): The outer list contains the
            tracking results of each frame in one video. The ndarray has two
            cases:
                - bbox: denotes the normal tracking box in [x1, y1, w, h]
                    format.
                - special tracking state: [0] denotes the unknown state,
                    namely the skipping frame after failure, [1] denotes the
                    initialized state, and [2] denotes the failed state.
        burnin: number of frames that have to be ignored after the
            re-initialization when calculating accuracy. Default is 10.
        ignore_unknown (bool): whether ignore the skipping frames after
            failures when calculating accuracy. Default is True.
        video_wh: bounding region (width, height)

    Return:
        Float: accuracy over the sequence.
    """
    pred_traj_region = trajectory2region(pred_trajectory)
    gt_traj_region = trajectory2region(gt_trajectory)
    overlaps = np.array(
        calculate_region_overlaps(pred_traj_region, gt_traj_region, video_wh))
    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(pred_traj_region):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(pred_traj_region), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False
    return np.mean(overlaps[mask]) if any(mask) else 0.


def eval_sot_accuracy_robustness(results,
                                 annotations,
                                 burnin=10,
                                 ignore_unknown=True,
                                 videos_wh=None):
    """Calculate accuracy and robustness over all tracking sequences.

    Args:
        results (list[list[ndarray]]): The first list contains the
            tracking results of each video. The second list contains the
            tracking results of each frame in one video. The ndarray have two
            cases:
                - bbox: denotes the normal tracking box in [x1, y1, w, h]
                    format.
                - special tracking state: [0] denotes the unknown state,
                    namely the skipping frame after failure, [1] denotes the
                    initialized state, and [2] denotes the failed state.
        annotations (list[ndarray]): The list contains the gt_bboxes of each
            video. The ndarray is gt_bboxes of one video. It's in (N, 4) shape.
            Each bbox is in (x1, y1, w, h) format.
        burnin: number of frames that have to be ignored after the
            re-initialization when calculating accuracy. Default is 10.
        ignore_unknown (bool): whether ignore the skipping frames after
            failures when calculating accuracy. Default is True.
        videos_wh (list[tuple(width, height), ...]): The list contains the
            width and height of each video. Default is None.

    Return:
        dict{str: float}: accuracy and robustness in EAO evaluation metric.
    """
    if vot is None:
        raise ImportError(
            'Please run'
            'pip install git+https://github.com/votchallenge/toolkit.git'
            'to manually install vot-toolkit')

    accuracy = 0
    num_fails = 0
    weight = 0
    for i, (gt_traj, pred_traj) in enumerate(zip(annotations, results)):
        assert len(gt_traj) == len(pred_traj)
        assert len(pred_traj[0]) == 1 and pred_traj[0][0] == 1
        num_fails += count_failures(pred_traj)
        accuracy += calc_accuracy(
            gt_traj,
            pred_traj,
            burnin=burnin,
            ignore_unknown=ignore_unknown,
            video_wh=videos_wh[i]) * len(pred_traj)
        weight += len(pred_traj)

    accuracy /= weight
    robustness = num_fails / weight * 100
    return dict(accuracy=accuracy, robustness=robustness, num_fails=num_fails)


def calc_eao_curve(overlaps, successes):
    """Calculate EAO curve over all tracking sequences.

    Args:
        overlaps (list[list]): The outer list contains the overlaps of each
            video. The inner list contains the overlap of each frame in one
            video.
        successes (list): The list contains the tracking states of last frame
            in each fragment.

    Return:
        ndarray: The N-th element in ndarray denotes the average overlaps from
            1 to N in all fragments.
    """
    max_length = max([len(_) for _ in overlaps])
    total_runs = len(overlaps)

    overlaps_array = np.zeros((total_runs, max_length), dtype=np.float32)
    # mask out frames which are not considered in EAO calculation. initial
    # value are zero, meaning ignored.
    mask = np.zeros((total_runs, max_length), dtype=np.float32)
    for i, (overlap, success) in enumerate(zip(overlaps, successes)):
        overlaps_array[i, :len(overlap)] = np.array(overlap)
        if not success:
            # tracker has failed during this sequence - consider all of
            # 'overlaps_array' and use the default padding from the end of
            # sequence to max length.
            mask[i, :] = 1
        else:
            # tracker has successfully tracked to the end - consider only this
            # part of the true sequence, and ignore the padding from the end of
            # sequence to max length.
            mask[i, :len(overlap)] = 1

    overlaps_array_sum = overlaps_array.copy()
    # overlaps_array_sum[i,j] means the mean overlap from 1 to j in i-th
    # sequence
    for j in range(1, overlaps_array_sum.shape[1]):
        overlaps_array_sum[:, j] = np.mean(overlaps_array[:, 1:j + 1], axis=1)

    return np.sum(overlaps_array_sum * mask, axis=0) / np.sum(mask, axis=0)


def eval_sot_eao(results, annotations, interval=[100, 356], videos_wh=None):
    """Calculate EAO socre over all tracking sequences.

    Args:
        results (list[list[ndarray]]): The first list contains the
            tracking results of each video. The second list contains the
            tracking results of each frame in one video. The ndarray have two
            cases:
                - bbox: denotes the normal tracking box in [x1, y1, w, h]
                    format.
                - special tracking state: [0] denotes the unknown state,
                    namely the skipping frame after failure, [1] denotes the
                    initialized state, and [2] denotes the failed state.
        annotations (list[ndarray]): The list contains the gt_bboxes of each
            video. The ndarray is gt_bboxes of one video. It's in (N, 4) shape.
            Each bbox is in (x1, y1, w, h) format.
        interval: an specified interval in EAO curve used to calculate the EAO
            score. There are different settings in different VOT challenge.
            Default is VOT2018 setting: [100, 356].
        videos_wh (list[tuple(width, height), ...]): The list contains the
            width and height of each video. Default is None.

    Return:
        dict[str, float]: EAO score in EAO evaluation metric.
    """
    if vot is None:
        raise ImportError(
            'Please run'
            'pip install git+https://github.com/votchallenge/toolkit.git'
            'to manually install vot-toolkit')

    if videos_wh is None:
        videos_wh = [None] * len(annotations)

    all_overlaps = []
    all_successes = []

    for i, (gt_traj, pred_traj) in enumerate(zip(annotations, results)):
        assert len(gt_traj) == len(
            pred_traj), f'{len(gt_traj)} == {len(pred_traj)}'
        # initialized bbox annotation is [1]
        assert len(pred_traj[0]) == 1 and pred_traj[0][
            0] == 1, f'{len(pred_traj[0])} == 1 and {pred_traj[0][0]} == 1'
        fail_inds, init_inds = locate_failures_inits(pred_traj)

        pred_traj = trajectory2region(pred_traj)
        gt_traj = trajectory2region(gt_traj)
        overlaps = calculate_region_overlaps(pred_traj, gt_traj, videos_wh[i])

        if len(fail_inds) > 0:
            for i in range(len(fail_inds)):
                all_overlaps.append(overlaps[init_inds[i]:fail_inds[i]])
                all_successes.append(False)

            # handle last initialization
            if len(init_inds) > len(fail_inds):
                # tracker was initialized, but it has not failed until the end
                # of the sequence
                all_overlaps.append(overlaps[init_inds[-1]:])
                all_successes.append(True)
        else:
            all_overlaps.append(overlaps)
            all_successes.append(True)

    eao_curve = calc_eao_curve(all_overlaps, all_successes)
    eao_score = np.mean(eao_curve[interval[0]:interval[1] + 1])
    eao = dict(eao=eao_score)
    return eao
