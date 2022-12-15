# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import mmcv
import numpy as np
import seaborn as sns
from mmdet.structures.mask import bitmap_to_polygon
from mmdet.visualization import DetLocalVisualizer as MMDET_DetLocalVisualizer
from mmdet.visualization.palette import _get_adaptive_scales
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer

from mmtrack.registry import VISUALIZERS
from mmtrack.structures import TrackDataSample


def random_color(seed):
    """Random a color according to the input seed."""
    np.random.seed(seed)
    colors = sns.color_palette()
    color = colors[np.random.choice(range(len(colors)))]
    color = tuple([int(255 * c) for c in color])
    return color


@VISUALIZERS.register_module()
class TrackLocalVisualizer(Visualizer):
    """MMTracking Local Visualizer for the MOT, VIS, SOT, VOS tasks.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
                Defaults to 0.8.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8):
        super().__init__(name, image, vis_backends, save_dir)
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `TrackLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances(self, image: np.ndarray,
                        instances: InstanceData) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)
        classes = self.dataset_meta.get('classes', None)

        # get colors and texts
        if hasattr(instances, 'instances_id'):
            # for the MOT and VIS tasks
            colors = [random_color(_id) for _id in instances.instances_id]
            categories = [
                classes[label] if classes is not None else f'cls{label}'
                for label in instances.labels
            ]
            if 'scores' in instances:
                texts = [
                    f'{category_name}\n{instance_id} | {score:.2f}'
                    for category_name, instance_id, score in zip(
                        categories, instances.instances_id, instances.scores)
                ]
            else:
                texts = [
                    f'{category_name}\n{instance_id}'
                    for category_name, instance_id in zip(
                        categories, instances.instances_id)
                ]
        else:
            # for the SOT and VOS tasks
            num_instances = max(
                len(instances.get('bboxes', [])),
                len(instances.get('masks', [])))
            colors = [random_color(_) for _ in range(num_instances)]
            if 'scores' in instances:
                texts = [f'{score:.2f}' for score in instances.scores]
            else:
                texts = None

        # draw bboxes and texts
        if 'bboxes' in instances:
            # draw bboxes
            bboxes = instances.bboxes.clone()
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)
            # draw texts
            if texts is not None:
                positions = bboxes[:, :2] + self.line_width
                areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                    bboxes[:, 2] - bboxes[:, 0])
                scales = _get_adaptive_scales(areas.cpu().numpy())
                for i, pos in enumerate(positions):
                    self.draw_texts(
                        texts[i],
                        pos,
                        colors='black',
                        font_sizes=int(13 * scales[i]),
                        bboxes=[{
                            'facecolor': [c / 255 for c in colors[i]],
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])

        # draw masks
        if 'masks' in instances:
            masks = instances.masks
            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['TrackDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: int = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`TrackDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT TrackDataSample.
                Default to True.
            draw_pred (bool): Whether to draw Prediction DTrackDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (int): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            assert 'gt_instances' in data_sample
            gt_img_data = self._draw_instances(image, data_sample.gt_instances)

        if draw_pred and data_sample is not None:
            assert 'pred_track_instances' in data_sample
            pred_instances = data_sample.pred_track_instances
            if 'scores' in pred_instances:
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr].cpu()
            pred_img_data = self._draw_instances(image, pred_instances)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, drawn_img, step)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)


@VISUALIZERS.register_module()
class DetLocalVisualizer(MMDET_DetLocalVisualizer):
    """MMTracking Local Visualizer for the VID task."""

    def add_datasample(self, *args, **kwargs):
        """Draw datasample and save to all backends."""
        if 'data_sample' in kwargs:
            # assign `pred_det_instances` to `pred_instances`
            # to hack the interface of `super().add_datasample()`.
            if 'pred_det_instances' in kwargs['data_sample']:
                kwargs['data_sample'].pred_instances = \
                    kwargs['data_sample'].pred_det_instances
        super().add_datasample(*args, **kwargs)
