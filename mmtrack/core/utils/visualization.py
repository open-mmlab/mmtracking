import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle


def random_color(seed):
    """Random a color according to the input seed."""
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color


def imshow_tracks(*args, backend='cv2', **kwargs):
    """Show the tracks on the input image."""
    if backend == 'cv2':
        return _cv2_show_tracks(*args, **kwargs)
    elif backend == 'plt':
        return _plt_show_tracks(*args, **kwargs)
    else:
        raise NotImplementedError()


def _cv2_show_tracks(img,
                     bboxes,
                     labels,
                     ids,
                     classes=None,
                     thickness=2,
                     font_scale=0.4,
                     show=False,
                     wait_time=0,
                     out_file=None):
    """Show the tracks with opencv."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 5
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    text_width, text_height = 10, 15
    for bbox, label, id in zip(bboxes, labels, ids):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = random_color(id)
        bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # id
        text = str(id)
        width = len(text) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            str(id), (x1, y1 + text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # score
        text = '{:.02f}'.format(score)
        width = len(text) * text_width
        img[y1 - text_height:y1, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def _plt_show_tracks(img,
                     bboxes,
                     labels,
                     ids,
                     classes=None,
                     thickness=1,
                     font_scale=0.5,
                     show=False,
                     wait_time=0,
                     out_file=None):
    """Show the tracks with matplotlib."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    assert bboxes.shape[0] == ids.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if isinstance(img, str):
        img = plt.imread(img)
    else:
        img = mmcv.bgr2rgb(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    if not show:
        matplotlib.use('Agg')

    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.autoscale(False)
    plt.subplots_adjust(
        top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.rcParams['figure.figsize'] = img_shape[1], img_shape[0]

    text_width, text_height = 12, 16
    for bbox, label, id in zip(bboxes, labels, ids):
        x1, y1, x2, y2, score = bbox
        w, h = int(x2 - x1), int(y2 - y1)
        left_top = (int(x1), int(y1))

        # bbox
        color = random_color(id)
        plt.gca().add_patch(
            Rectangle(
                left_top, w, h, thickness, edgecolor=color, facecolor='none'))

        # id
        text = str(id)
        width = len(text) * text_width
        plt.gca().add_patch(
            Rectangle((left_top[0], left_top[1]),
                      width,
                      text_height,
                      thickness,
                      edgecolor=color,
                      facecolor=color))
        plt.text(left_top[0], left_top[1] + text_height + 2, text, fontsize=5)

    if out_file is not None:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)

    if show:
        plt.draw()
        plt.pause(wait_time / 1000.)
    else:
        plt.show()
    plt.clf()
    return img
