from collections import defaultdict


def track2result(bboxes, labels, ids):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds].cpu().numpy()
    labels = labels[valid_inds].cpu().numpy()
    ids = ids[valid_inds].cpu().numpy()

    outputs = defaultdict(list)
    for bbox, label, id in zip(bboxes, labels, ids):
        outputs[id] = dict(bbox=bbox, label=label)
    return outputs


# def result2track(track_result):
#     track_bboxes = np.zeros((0, 5))
#     track_ids = np.zeros((0))
#     for k, v in track_results.items():
#         track_bboxes = np.concatenate((track_bboxes, v['bbox'][None, :]),
#                                       axis=0)
#         track_ids = np.concatenate((track_ids, np.array([k])), axis=0)
#     return track_bboxes, track_ids
