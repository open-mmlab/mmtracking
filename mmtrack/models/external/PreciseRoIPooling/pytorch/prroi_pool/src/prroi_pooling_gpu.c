/*
 * File   : prroi_pooling_gpu.c
 * Author : Jiayuan Mao, Tete Xiao
 * Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
 * Date   : 07/13/2018
 *
 * Distributed under terms of the MIT license.
 * Copyright (c) 2017 Megvii Technology Limited.
 */

#include <math.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>

#include "prroi_pooling_gpu_impl.cuh"


at::Tensor prroi_pooling_forward_cuda(const at::Tensor &features, const at::Tensor &rois, int pooled_height, int pooled_width, float spatial_scale) {
    int nr_rois = rois.size(0);
    int nr_channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);
    int top_count = nr_rois * nr_channels * pooled_height * pooled_width;
    auto output = at::zeros({nr_rois, nr_channels, pooled_height, pooled_width}, features.options());

    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return output;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    PrRoIPoolingForwardGpu(
        stream, features.data<float>(), rois.data<float>(), output.data<float>(),
        nr_channels, height, width, pooled_height, pooled_width, spatial_scale,
        top_count
    );

    THCudaCheck(cudaGetLastError());
    return output;
}

at::Tensor prroi_pooling_backward_cuda(
    const at::Tensor &features, const at::Tensor &rois, const at::Tensor &output, const at::Tensor &output_diff,
    int pooled_height, int pooled_width, float spatial_scale) {

    auto features_diff = at::zeros_like(features);

    int nr_rois = rois.size(0);
    int batch_size = features.size(0);
    int nr_channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);
    int top_count = nr_rois * nr_channels * pooled_height * pooled_width;
    int bottom_count = batch_size * nr_channels * height * width;

    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return features_diff;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    PrRoIPoolingBackwardGpu(
        stream,
        features.data<float>(), rois.data<float>(), output.data<float>(), output_diff.data<float>(),
        features_diff.data<float>(),
        nr_channels, height, width, pooled_height, pooled_width, spatial_scale,
        top_count, bottom_count
    );

    THCudaCheck(cudaGetLastError());
    return features_diff;
}

at::Tensor prroi_pooling_coor_backward_cuda(
    const at::Tensor &features, const at::Tensor &rois, const at::Tensor &output, const at::Tensor &output_diff,
    int pooled_height, int pooled_width, float spatial_scale) {

    auto coor_diff = at::zeros_like(rois);

    int nr_rois = rois.size(0);
    int nr_channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);
    int top_count = nr_rois * nr_channels * pooled_height * pooled_width;
    int bottom_count = nr_rois * 5;

    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return coor_diff;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    PrRoIPoolingCoorBackwardGpu(
        stream,
        features.data<float>(), rois.data<float>(), output.data<float>(), output_diff.data<float>(),
        coor_diff.data<float>(),
        nr_channels, height, width, pooled_height, pooled_width, spatial_scale,
        top_count, bottom_count
    );

    THCudaCheck(cudaGetLastError());
    return coor_diff;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prroi_pooling_forward_cuda", &prroi_pooling_forward_cuda, "PRRoIPooling_forward");
    m.def("prroi_pooling_backward_cuda", &prroi_pooling_backward_cuda, "PRRoIPooling_backward");
    m.def("prroi_pooling_coor_backward_cuda", &prroi_pooling_coor_backward_cuda, "PRRoIPooling_backward_coor");
}
