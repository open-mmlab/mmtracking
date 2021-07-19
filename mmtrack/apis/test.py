import os
import os.path as osp
import shutil
import tempfile
import time
from collections import defaultdict

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.utils import print_log

from mmtrack.datasets import MOTChallengeDataset


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    out_video=False,
                    out_image=False,
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
        out_video (bool, optional): Whether to output video.
            Defaults to False.
        out_image (bool, optional): Whether to output image.
            Defaults to False.
        fps (int, optional): FPS of the output video.
            Defaults to 3.
        show_score_thr (float, optional): The score threshold of visualization.
            (Not supported for now). Defaults to 0.3.

    Returns:
        dict[str, list]: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    last_video_name = None
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        for k, v in result.items():
            results[k].append(v)

        if show or out_dir:
            # TODO: support SOT and VOD visualization
            if isinstance(dataset, MOTChallengeDataset):
                img_path = data['img_metas'][0].data[0][0]['filename']
                video_name = img_path.split('/')[-3]
                mot_visualization(img_path, video_name,
                                  result['track_results'],
                                  out_dir, model, i, last_video_name,
                                  len(dataset), show, out_video, out_image,
                                  fps)
                last_video_name = video_name
            else:
                raise NotImplementedError(
                    'Only support multiple object tracking visualization now.')

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
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


def mot_visualization(img_path,
                      video_name,
                      result,
                      out_dir,
                      model,
                      idx=0,
                      last_video_name=None,
                      dataset_len=0,
                      show=False,
                      out_video=False,
                      out_image=False,
                      fps=3):
    """Visualize multiple object tracking results.

    Args:
        img_path (str): Path of image to be displayed.
        video_name (str): Video name of the image.
        result (ndarray): Testing result of the image.
        out_dir (str): Path of directory to save the visualization results.
        model (nn.Module): Model to be tested.
        idx (int, optional): Index of the image. Defaults to 0.
        last_video_name (str, optional): Video name of the last image.
            Defaults to None.
        dataset_len (int, optional): Length of dataset. Defaults to 0.
        show (bool, optional):
        out_video (bool, optional): Whether to output video.
            Defaults to False.
        out_image (bool, optional): Whether to output image.
            Defaults to False.
        fps (int, optional): FPS of the output video.
            Defaults to 3.

    Returns:
        vis_frame (ndarray): Visualized image.
    """
    assert isinstance(img_path, str)
    frame_id = int(img_path.split('/')[-1].split('.')[0])
    out_file = osp.join(
        out_dir, f'{video_name}/{frame_id:06d}.jpg') if out_dir else None
    vis_frame = model.module.show_result(
        img_path, result, show=show, out_file=out_file)
    if last_video_name != video_name and idx or idx == dataset_len - 1:
        if out_video:
            imgs_dir = osp.join(out_dir, last_video_name)
            start_frame_id = int(sorted(os.listdir(imgs_dir))[0].split('.')[0])
            end_frame_id = int(sorted(os.listdir(imgs_dir))[-1].split('.')[0])
            print_log(f'Start processing video {last_video_name}')
            mmcv.frames2video(
                imgs_dir,
                f'{imgs_dir}.mp4',
                fps=fps,
                fourcc='mp4v',
                start=start_frame_id,
                end=end_frame_id)
        if not out_image:
            shutil.rmtree(osp.join(out_dir, last_video_name))
    return vis_frame
