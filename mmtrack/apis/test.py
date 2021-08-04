import os
import os.path as osp
import shutil
import tempfile
import time
from collections import defaultdict

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmtrack.datasets import ImagenetVIDDataset


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    fps=3,
                    show_score_thr=0.3):
    """Test model with single gpu.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): If True, visualize the prediction results.
            Defaults to False.
        out_dir (str, optional): Path of directory to save the
            visualization results. Defaults to None.
        fps (int, optional): FPS of the output video.
            Defaults to 3.
        show_score_thr (float, optional): The score threshold of visualization.
            Defaults to 0.3.

    Returns:
        dict[str, list]: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        for k, v in result.items():
            results[k].append(v)

        batch_size = data['img'][0].size(0)
        if show or out_dir:
            assert batch_size == 1, 'Only support batch_size=1 when testing.'
            img_tensor = data['img'][0]
            img_meta = data['img_metas'][0].data[0][0]
            img = tensor2imgs(img_tensor, **img_meta['img_norm_cfg'])[0]

            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            if out_dir:
                out_file = osp.join(out_dir, img_meta['ori_filename'])
            else:
                out_file = None

            result_show = result[
                'track_results'] if 'track_results' in result else result
            model.module.show_result(
                img_show,
                result_show,
                show=show,
                out_file=out_file,
                score_thr=show_score_thr)

        for _ in range(batch_size):
            prog_bar.update()

    if out_dir:
        img_name = img_meta['ori_filename'].rsplit('/', 1)[1]
        img_idx, img_fmt = img_name.split('.')
        filename_tmpl = '{:0' + str(len(img_idx)) + 'd}.' + img_fmt
        if isinstance(dataset, ImagenetVIDDataset):
            out_dir = osp.join(out_dir, os.listdir(out_dir)[0])
        video_names = os.listdir(out_dir)
        for video_name in video_names:
            img_dirs = osp.join(out_dir, video_name)
            if not os.listdir(img_dirs)[0].endswith(f'.{img_fmt}'):
                img_dirs = f'{img_dirs}/{os.listdir(img_dirs)[0]}'
            img_names = sorted(os.listdir(img_dirs))
            start_frame_id = int(img_names[0].split('.')[0])
            end_frame_id = int(img_names[-1].split('.')[0])

            mmcv.frames2video(
                img_dirs,
                f'{out_dir}/{video_name}.mp4',
                fps=fps,
                fourcc='mp4v',
                filename_tmpl=filename_tmpl,
                start=start_frame_id,
                end=end_frame_id,
                show_progress=False)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker. 'gpu_collect=True' is not
    supported for now.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Defaults to None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Defaults to False.

    Returns:
        dict[str, list]: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        for k, v in result.items():
            results[k].append(v)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        raise NotImplementedError
    else:
        results = collect_results_cpu(results, tmpdir)
    return results


def collect_results_cpu(result_part, tmpdir=None):
    """Collect results on cpu mode.

    Saves the results on different gpus to 'tmpdir' and collects them by the
    rank 0 worker.

    Args:
        result_part (dict[list]): The part of prediction results.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. If is None, use `tempfile.mkdtemp()`
            to make a temporary path. Defaults to None.

    Returns:
        dict[str, list]: The prediction results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = defaultdict(list)
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_file = mmcv.load(part_file)
            for k, v in part_file.items():
                part_list[k].extend(v)
        shutil.rmtree(tmpdir)
        return part_list
