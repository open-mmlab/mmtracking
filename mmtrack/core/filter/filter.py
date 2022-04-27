# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def apply_filter(feat, filter, dilation_factors=None):
    """Applies the filter on the input features (feat).

    The number of groups is
        automatically calculated.
    args:
        feat: These are the input features. Must have dimensions
            (images_in_sequence, sequences, feat_dim, H, W)
        filter: The filter to apply. Must have dimensions
            (sequences, feat_dim, fH, fW) or
            (sequences, filters, feat_dim/groups, fH, fW)
    output:
        scores: Output of filtering. Dimensions
            (images_in_sequence, sequences, yH, yW) or
            (images_in_sequence, sequences, filters, yH, yW)
    """

    multiple_filters = (filter.dim() == 5)

    padding = (filter.shape[-2] // 2, filter.shape[-1] // 2)

    num_images = feat.shape[0]
    num_sequences = feat.shape[1] if feat.dim() == 5 else 1
    num_filters = filter.shape[1] if multiple_filters else 1
    num_channels = feat.shape[-3]
    groups = num_channels // filter.shape[-3]

    assert num_filters % groups == 0 and num_channels % groups == 0

    if multiple_filters:
        if dilation_factors is None:
            scores = F.conv2d(
                feat.reshape(num_images, -1, feat.shape[-2], feat.shape[-1]),
                filter.view(-1, *filter.shape[-3:]),
                padding=padding,
                groups=num_sequences * groups)

            return scores.view(num_images, num_sequences, -1, scores.shape[-2],
                               scores.shape[-1])
        else:
            scores_all = []
            start_id = 0

            for d_factor, num_filters_with_d in dilation_factors.items():
                f_d = filter[:, start_id:start_id + num_filters_with_d,
                             ...].contiguous()

                padding_d = [p + d_factor - 1 for p in padding]
                scores_d = F.conv2d(
                    feat.reshape(num_images, -1, feat.shape[-2],
                                 feat.shape[-1]),
                    f_d.view(-1, *f_d.shape[-3:]),
                    padding=padding_d,
                    groups=num_sequences * groups,
                    dilation=d_factor)
                scores_d = scores_d.view(num_images, num_sequences, -1,
                                         scores_d.shape[-2],
                                         scores_d.shape[-1])
                scores_all.append(scores_d)
                start_id += num_filters_with_d

            scores = torch.cat(scores_all, dim=2)
            return scores

    scores = F.conv2d(
        feat.reshape(num_images, -1, feat.shape[-2], feat.shape[-1]),
        filter,
        padding=padding,
        groups=num_sequences)

    return scores.view(num_images, num_sequences, scores.shape[-2],
                       scores.shape[-1])


def apply_feat_transpose(feat, input, filter_ksz, training=True, groups=1):
    """Applies the transposed operation off apply_filter w.r.t.

    filter itself.
        Can be used to compute the filter gradient.
    args:
        feat: These are the input features. Must have dimensions
            (images_in_sequence, sequences, feat_dim, H, W)
        input: Input activation (e.g. residuals). Must have dimensions
            (images_in_sequence, sequences, yH, yW) or
            (images_in_sequence, sequences, filters, yH, yW)
        training: Choose the faster implementation whether training or not.
    output:
        Output of transposed operation. Dimensions
            (sequences, feat_dim, fH, fW)
    """

    if groups != 1:
        raise NotImplementedError('Not implemented other values of group.')

    if training or input.dim() == 5:
        return _apply_feat_transpose_v3(feat, input, filter_ksz)
    return _apply_feat_transpose_v2(feat, input, filter_ksz)


def _apply_feat_transpose_v2(feat, input, filter_ksz):
    """Fast forward and slow backward."""

    multiple_filters = (input.dim() == 5)

    num_images = feat.shape[0]
    num_sequences = feat.shape[1] if feat.dim() == 5 else 1
    num_filters = input.shape[2] if multiple_filters else 1
    if isinstance(filter_ksz, int):
        filter_ksz = (filter_ksz, filter_ksz)

    trans_pad = [(ksz - 1) // 2 for ksz in filter_ksz]

    if multiple_filters:
        filter_grad = F.conv2d(
            input.reshape(-1, num_filters, input.shape[-2],
                          input.shape[-1]).permute(1, 0, 2, 3),
            feat.reshape(-1, 1, feat.shape[-2], feat.shape[-1]),
            padding=trans_pad,
            groups=num_images * num_sequences)

        if num_images == 1:
            return filter_grad.view(num_filters, num_sequences, -1,
                                    filter_grad.shape[-2],
                                    filter_grad.shape[-1]).flip(
                                        (3, 4)).permute(1, 0, 2, 3, 4)
        return filter_grad.view(num_filters, num_images, num_sequences, -1,
                                filter_grad.shape[-2],
                                filter_grad.shape[-1]).sum(dim=1).flip(
                                    (3, 4)).permute(1, 0, 2, 3, 4)

    filter_grad = F.conv2d(
        input.reshape(1, -1, input.shape[-2], input.shape[-1]),
        feat.reshape(-1, 1, feat.shape[-2], feat.shape[-1]),
        padding=trans_pad,
        groups=num_images * num_sequences)

    return filter_grad.view(num_images, num_sequences, -1,
                            filter_grad.shape[-2],
                            filter_grad.shape[-1]).sum(dim=0).flip((2, 3))


def _apply_feat_transpose_v3(feat, input, filter_ksz):
    """Slow forward fast backward."""

    multiple_filters = (input.dim() == 5)

    num_images = feat.shape[0]
    num_sequences = feat.shape[1] if feat.dim() == 5 else 1
    num_filters = input.shape[2] if multiple_filters else 1
    if isinstance(filter_ksz, int):
        filter_ksz = (filter_ksz, filter_ksz)

    trans_pad = [ksz // 2 for ksz in filter_ksz]

    filter_grad = F.conv2d(
        feat.reshape(-1, feat.shape[-3], feat.shape[-2],
                     feat.shape[-1]).permute(1, 0, 2, 3),
        input.reshape(-1, 1, input.shape[-2], input.shape[-1]),
        padding=trans_pad,
        groups=num_images * num_sequences)

    if multiple_filters:
        if num_images == 1:
            return filter_grad.view(-1, num_sequences, num_filters,
                                    filter_grad.shape[-2],
                                    filter_grad.shape[-1]).permute(
                                        1, 2, 0, 3, 4)
        return filter_grad.view(-1, num_images, num_sequences, num_filters,
                                filter_grad.shape[-2],
                                filter_grad.shape[-1]).sum(dim=1).permute(
                                    1, 2, 0, 3, 4)

    if num_images == 1:
        return filter_grad.permute(1, 0, 2, 3)
    return filter_grad.view(-1, num_images, num_sequences,
                            filter_grad.shape[-2],
                            filter_grad.shape[-1]).sum(dim=1).permute(
                                1, 0, 2, 3)


def _apply_feat_transpose_v4(feat, input, filter_ksz):
    """Slow forward fast backward."""

    num_sequences = feat.shape[1] if feat.dim() == 5 else 1
    if isinstance(filter_ksz, int):
        filter_ksz = (filter_ksz, filter_ksz)

    trans_pad = [ksz // 2 for ksz in filter_ksz]

    filter_grad = F.conv2d(
        feat.permute(2, 1, 0, 3, 4).reshape(feat.shape[-3], -1, feat.shape[-2],
                                            feat.shape[-1]),
        input.permute(1, 0, 2, 3),
        padding=trans_pad,
        groups=num_sequences)

    return filter_grad.permute(1, 0, 2, 3)


def filter_gradient(feat, filter, label=None, training=True):
    """Computes gradient of the filter when applied on the input features and
    ground truth label.

    args:
        feat: These are the input features. Must have dimensions
            (images_in_sequence, sequences, feat_dim, H, W)
        filter: The filter to apply. Must have dimensions
            (sequences, feat_dim, fH, fW)
        label: Ground truth label in the L2 loss. Dimensions
            (images_in_sequence, sequences, yH, yW)
    output:
        filter_gradient: Dimensions same as input filter
            (sequences, feat_dim, fH, fW)
    """

    residuals = apply_filter(feat, filter)
    if label is not None:
        residuals = residuals - label
    filter_ksz = (filter.shape[-2], filter.shape[-1])
    return apply_feat_transpose(feat, residuals, filter_ksz, training=training)
