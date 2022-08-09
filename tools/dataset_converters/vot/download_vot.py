# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import socket
import zipfile
from urllib import error, request

from tqdm import tqdm

try:
    from vot.utilities import extract_files
    from vot.utilities.net import (download_json, download_uncompress,
                                   get_base_url, join_url)
except ImportError:
    raise ImportError(
        'Please run'
        'pip install git+https://github.com/votchallenge/toolkit.git'
        'to manually install vot-toolkit')

VOT_DATASETS = dict(
    vot2018='http://data.votchallenge.net/vot2018/main/description.json',
    vot2018_lt=  # noqa: E251
    'http://data.votchallenge.net/vot2018/longterm/description.json',
    vot2019='http://data.votchallenge.net/vot2019/main/description.json',
    vot2019_lt=  # noqa: E251
    'http://data.votchallenge.net/vot2019/longterm/description.json',
    vot2019_rgbd='http://data.votchallenge.net/vot2019/rgbd/description.json',
    vot2019_rgbt=  # noqa: E251
    'http://data.votchallenge.net/vot2019/rgbtir/meta/description.json',
    vot2020='https://data.votchallenge.net/vot2020/shortterm/description.json',
    vot2020_rgbt=  # noqa: E251
    'http://data.votchallenge.net/vot2020/rgbtir/meta/description.json',
    vot2021='https://data.votchallenge.net/vot2021/shortterm/description.json')


def download_dataset(dataset_name, path):
    url = VOT_DATASETS[dataset_name]
    meta = download_json(url)
    base_url = get_base_url(url) + '/'
    for sequence in tqdm(meta['sequences']):
        sequence_directory = os.path.join(path, sequence['name'])
        os.makedirs(sequence_directory, exist_ok=True)

        annotations_url = join_url(base_url, sequence['annotations']['url'])
        download_uncompress(annotations_url, sequence_directory)

        for cname, channel in sequence['channels'].items():
            channel_directory = os.path.join(sequence_directory, cname)
            os.makedirs(channel_directory, exist_ok=True)
            channel_url = join_url(base_url, channel['url'])
            tmp_zip = osp.join(channel_directory, f'{sequence["name"]}.zip')
            download_url(channel_url, tmp_zip)
            try:
                extract_files(tmp_zip, channel_directory)
            except zipfile.BadZipFile:
                print(f'[Error]: Please download {sequence["name"]} video \
                        manually through the {channel_url}')
            os.remove(tmp_zip)


def download_url(url, saved_file):
    video_zip = osp.basename(saved_file)
    try:
        request.urlretrieve(url, saved_file)
    except error.HTTPError as e:
        print(e)
        print('\r\n' + url + ' download failed!' + '\r\n')
    except (error.ContentTooShortError, socket.timeout):
        count = 1
        while count <= 5:
            try:
                request.urlretrieve(url, saved_file)
                break
            except (error.ContentTooShortError, socket.timeout):
                err_info = 'ReDownloading %s for %d time' % (video_zip, count)
                print(err_info)
                count += 1
        if count > 5:
            print('downloading %s failed!' % video_zip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--dataset_name',
        help='dataset name',
        default='vot2018',
        choices=[
            'vot2018',
            'vot2018_lt',
            'vot2019',
            'vot2019_lt',
            'vot2019_rgbd',
            'vot2019_rgbt',
            'vot2020',
            'vot2020_rgbt',
            'vot2021',
        ],
    )
    parser.add_argument('--save_path', help='dataset saved path', default='./')
    args = parser.parse_args()
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    download_dataset(args.dataset_name, args.save_path)
