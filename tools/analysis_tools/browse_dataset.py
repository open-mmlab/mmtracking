# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmengine
import numpy as np
from mmdet.models.utils import mask2ndarray
from mmengine import Config, DictAction
from mmengine.structures import InstanceData

from mmtrack.registry import DATASETS, VISUALIZERS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmtrack into the registries
    register_all_modules(init_default_scope=True)

    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = mmengine.ProgressBar(len(dataset))
    gt_sample = TrackDataSample()  # just to wrap the `gt_instances`
    for idx, item in enumerate(dataset):
        data_sample = item['data_samples']
        for img_key, imgs in item['inputs'].items():
            img_paths = data_sample.get(img_key + '_path')
            img_key_prefix = img_key[:-3]
            gt_instances = data_sample.get(img_key_prefix + 'gt_instances')
            if not isinstance(img_paths, list):
                img_paths = [img_paths]
            for img_idx in range(imgs.shape[0]):
                new_gt_instances = InstanceData()
                img_path = img_paths[img_idx]
                img = imgs[img_idx].permute(1, 2, 0).numpy()
                # For each item, their file names may be the same.
                # Create a new folder to avoid overwriting the image files.
                out_file = osp.join(args.output_dir,
                                    str(idx).zfill(6),
                                    f'{img_key_prefix}img_{img_idx}.jpg'
                                    ) if args.output_dir is not None else None

                img = img[..., [2, 1, 0]]  # bgr to rgb
                # Get the correct index for each instance by using
                # map_instances_to_img_idx
                map_instances_to_img_idx = gt_instances.\
                    map_instances_to_img_idx.numpy()
                idx_bool_flag = (map_instances_to_img_idx == img_idx)
                for key in ['bboxes', 'labels', 'instances_id']:
                    if key in gt_instances:
                        new_gt_instances[key] = gt_instances[key][
                            idx_bool_flag]

                gt_masks = gt_instances.get('masks', None)
                if gt_masks is not None:
                    gt_masks = gt_masks[idx_bool_flag]
                    masks = mask2ndarray(gt_masks)
                    new_gt_instances['masks'] = masks.astype(np.bool)

                gt_sample.gt_instances = new_gt_instances

                visualizer.add_datasample(
                    osp.basename(img_path),
                    img,
                    data_sample=gt_sample,
                    draw_pred=False,
                    show=not args.not_show,
                    wait_time=args.show_interval,
                    out_file=out_file)
                # Record file path mapping.
                if args.output_dir is not None:
                    with open(
                            osp.join(args.output_dir,
                                     str(idx).zfill(6), 'info.txt'), 'a') as f:
                        f.write(f'The source filepath of'
                                f' `{img_key_prefix}img_{img_idx}.jpg`'
                                f' is `{img_path}`.\n')

        progress_bar.update()


if __name__ == '__main__':
    main()
