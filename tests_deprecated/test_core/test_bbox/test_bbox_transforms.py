# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmtrack.core.bbox import (bbox_cxcyah_to_xyxy, bbox_cxcywh_to_x1y1wh,
                               bbox_xyxy_to_cxcyah, bbox_xyxy_to_x1y1wh,
                               quad2bbox)


def test_quad2bbox():
    quad = torch.zeros((5, 8), dtype=torch.float)
    low_coord_index = torch.tensor([0, 1, 3, 6], dtype=torch.long)
    high_coord_index = torch.tensor([2, 4, 5, 7], dtype=torch.long)
    quad[:, low_coord_index] = torch.randint(1, 10, (5, 4), dtype=torch.float)
    quad[:, high_coord_index] = torch.randint(
        10, 20, (5, 4), dtype=torch.float)
    bbox = quad2bbox(quad)
    assert (bbox > 0).all()


def test_bbox_cxcywh_to_x1y1wh():
    cx = torch.randint(1, 10, (5, 1), dtype=torch.float)
    cy = torch.randint(1, 10, (5, 1), dtype=torch.float)
    w = torch.randint(1, 10, (5, 1), dtype=torch.float)
    h = torch.randint(1, 10, (5, 1), dtype=torch.float)
    bbox = torch.cat((cx, cy, w, h), dim=-1)
    bbox_new = bbox_cxcywh_to_x1y1wh(bbox)
    assert (bbox_new[:, :2] < bbox[:, :2]).all()


def test_bbox_xyxy_to_x1y1wh():
    x1 = torch.randint(1, 10, (5, 1), dtype=torch.float)
    y1 = torch.randint(1, 10, (5, 1), dtype=torch.float)
    x2 = torch.randint(10, 20, (5, 1), dtype=torch.float)
    y2 = torch.randint(10, 20, (5, 1), dtype=torch.float)
    bbox = torch.cat((x1, y1, x2, y2), dim=-1)
    bbox_new = bbox_xyxy_to_x1y1wh(bbox)
    assert (bbox_new[:, 2:] > 0).all()


def test_bbox_xyxy_to_cxcyah():
    x1 = torch.randint(1, 10, (5, 1), dtype=torch.float)
    y1 = torch.randint(1, 10, (5, 1), dtype=torch.float)
    x2 = torch.randint(10, 20, (5, 1), dtype=torch.float)
    y2 = torch.randint(10, 20, (5, 1), dtype=torch.float)
    bbox = torch.cat((x1, y1, x2, y2), dim=-1)
    bbox_new = bbox_xyxy_to_cxcyah(bbox)
    assert (bbox_new > 0).all()


def test_bbox_cxcyah_to_xyxy():
    cx = torch.randint(1, 10, (5, 1), dtype=torch.float)
    cy = torch.randint(1, 10, (5, 1), dtype=torch.float)
    ratio = torch.randint(10, 20, (5, 1), dtype=torch.float)
    h = torch.randint(10, 20, (5, 1), dtype=torch.float)
    bbox = torch.cat((cx, cy, ratio, h), dim=-1)
    bbox_new = bbox_cxcyah_to_xyxy(bbox)
    assert bbox_new.shape == bbox.shape
