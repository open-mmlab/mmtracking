import numpy as np
import torch
from torch.nn import functional as F


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (Tensor or ndarray): 2-D feature matrix.
        input2 (Tensor or ndarray): 2-D feature matrix.
        metric (str, optional): 'euclidean', 'squared_euclidean' or 'cosine'.
            Default is 'euclidean'.
    Returns:
        Tensor or ndarray: distance matrix.
    """
    allowed_metrics = ['euclidean', 'squared_euclidean', 'cosine']
    assert metric in allowed_metrics, \
        f'Unknown distance metric: {metric}. ' \
        'Please choose either "euclidean", "squared_euclidean" or "cosine".'

    if isinstance(input1, torch.Tensor) and isinstance(input2, torch.Tensor):
        assert input1.dim(
        ) == 2, f'Expected 2-D tensor, but got {input1.dim()}-D'
        assert input2.dim(
        ) == 2, f'Expected 2-D tensor, but got {input2.dim()}-D'
        assert input1.size(1) == input2.size(1)

        if metric == 'euclidean':
            distmat = torch_euclidean_squared_distance(input1, input2).sqrt()
        elif metric == 'squared_euclidean':
            distmat = torch_euclidean_squared_distance(input1, input2)
        elif metric == 'cosine':
            distmat = torch_cosine_distance(input1, input2)
    elif isinstance(input1, np.ndarray) and isinstance(input2, np.ndarray):
        assert input1.ndim == 2, \
            f'Expected 2-D ndarray, but got {input1.ndim}-D'
        assert input2.ndim == 2, \
            f'Expected 2-D ndarray, but got {input2.ndim}-D'
        assert input1.shape[1] == input2.shape[1]

        if metric == 'euclidean':
            distmat = np.sqrt(
                ndarray_euclidean_squared_distance(input1, input2))
        elif metric == 'squared_euclidean':
            distmat = ndarray_euclidean_squared_distance(input1, input2)
        elif metric == 'cosine':
            distmat = ndarray_cosine_distance(input1, input2)
    else:
        raise TypeError(
            'Both input1 and input2 are expected to be tensor or ndarray,'
            f'but got {type(input1)} and {type(input2)}')

    return distmat


def torch_euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance between two tensor.

    Args:
        input1 (Tensor): 2-D feature matrix.
        input2 (Tensor): 2-D feature matrix.
    Returns:
        Tensor: distance matrix.
    """
    input1_squared = torch.pow(input1, 2).sum(dim=1, keepdim=True)
    input2_squared = input1_squared if input1 is input2 else torch.pow(
        input2, 2).sum(
            dim=1, keepdim=True)
    distmat = input1_squared + input2_squared.t() - 2 * torch.mm(
        input1, input2.t())
    return distmat


def torch_cosine_distance(input1, input2):
    """Computes cosine distance between two tensor.

    Args:
        input1 (Tensor): 2-D feature matrix.
        input2 (Tensor): 2-D feature matrix.
    Returns:
        Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = input1_normed if input1 is input2 else F.normalize(
        input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def ndarray_euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance between two ndarray.

    Args:
        input1 (ndarray): 2-D ndarray matrix.
        input2 (ndarray): 2-D ndarray matrix.
    Returns:
        ndarray: distance matrix.
    """
    input1_squared = np.sum(np.power(input1, 2), axis=1, keepdims=True)
    input2_squared = input1_squared if input1 is input2 else np.sum(
        np.power(input2, 2), axis=1, keepdims=True)
    distmat = input1_squared + input2_squared.T - 2 * np.dot(input1, input2.T)
    return distmat


def ndarray_cosine_distance(input1, input2):
    """Computes cosine squared distance between two ndarray.

    Args:
        input1 (ndarray): 2-D ndarray matrix.
        input2 (ndarray): 2-D ndarray matrix.
    Returns:
        ndarray: distance matrix.
    """
    input1_normed = input1 / np.linalg.norm(
        input1, ord=2, axis=1, keepdims=True)
    input2_normed = input1_normed if input1 is input2 \
        else input2 / np.linalg.norm(input2, ord=2, axis=1, keepdims=True)
    distmat = 1 - np.dot(input1_normed, input2_normed.T)
    return distmat
