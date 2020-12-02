import numpy as np
from mmdet.core.anchor import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module()
class SOTAnchorGenerator(object):

    def __init__(self,
                 strides=8,
                 ratios=[0.33, 0.5, 1, 2, 3],
                 scales=[8],
                 score_map_size=25,
                 **kwargs):
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        self.num_anchor = len(self.ratios) * len(self.scales)
        self.score_map_size = score_map_size
        self.anchors = self.gen_anchors(strides, ratios, scales,
                                        score_map_size)

        hanning = np.hanning(self.score_map_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.num_anchor)

    def gen_anchors(self, strides, ratios, scales, score_map_size):
        anchor = self.gen_one_location_anchors(strides, ratios, scales)

        x_grid, y_grid = np.meshgrid(
            [strides * i for i in range(score_map_size)],
            [strides * j for j in range(score_map_size)])

        num_anchor = len(scales) * len(ratios)
        x_grid = np.tile(x_grid.flatten(), num_anchor)
        y_grid = np.tile(y_grid.flatten(), num_anchor)

        anchor = np.tile(anchor, score_map_size * score_map_size).reshape(
            (-1, 4))
        offset = -(score_map_size // 2) * strides
        anchor[:, 0] = offset + x_grid.astype(np.float32)
        anchor[:, 1] = offset + y_grid.astype(np.float32)
        return anchor

    def gen_one_location_anchors(self, strides, ratios, scales):
        """generate anchors based on predefined configuration."""
        num_anchor = len(scales) * len(ratios)
        anchors = np.zeros((num_anchor, 4), dtype=np.float32)
        base_size = strides**2

        for i, ratio in enumerate(ratios):
            base_width = int(np.sqrt(base_size / float(ratio)))
            base_height = int(np.sqrt(base_size * ratio))
            for j, scale in enumerate(scales):
                width = scale * base_width
                height = scale * base_height
                anchors[i * len(scales) + j] = [0., 0., width, height]
        return anchors
