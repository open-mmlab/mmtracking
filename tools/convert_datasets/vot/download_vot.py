import argparse
import os
import os.path as osp
import socket
import zipfile
from urllib import error, request

from tqdm import tqdm
from vot.utilities import extract_files
from vot.utilities.net import (download_json, download_uncompress,
                               get_base_url, join_url)

VOT_DATASETS = {
    'vot-st2018':
    'http://data.votchallenge.net/vot2018/main/description.json',
    'vot-lt2018':
    'http://data.votchallenge.net/vot2018/longterm/description.json',
    'vot-st2019':
    'http://data.votchallenge.net/vot2019/main/description.json',
    'vot-lt2019':
    'http://data.votchallenge.net/vot2019/longterm/description.json',
    'vot-rgbd2019':
    'http://data.votchallenge.net/vot2019/rgbd/description.json',
    'vot-rgbt2019':
    'http://data.votchallenge.net/vot2019/rgbtir/meta/description.json',
    'vot-st2020':
    'https://data.votchallenge.net/vot2020/shortterm/description.json',
    'vot-rgbt2020':
    'http://data.votchallenge.net/vot2020/rgbtir/meta/description.json',
    'vot-st2021':
    'https://data.votchallenge.net/vot2021/shortterm/description.json',
}


def download(dataset_name, path):
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
    except error.ContentTooShortError:
        count = 1
        while count <= 5:
            try:
                request.urlretrieve(url, saved_file)
                break
            except error.ContentTooShortError:
                err_info = 'ReDownloading %s for %d time' % (video_zip, count)
                print(err_info)
                count += 1
        if count > 5:
            print('downloading %s failed!' % video_zip)
    except error.HTTPError as e:
        print(e)
        print('\r\n' + url + ' download failed!' + '\r\n')
    except socket.timeout:
        count = 1
        while count <= 5:
            try:
                request.urlretrieve(url, saved_file)
                break
            except socket.timeout:
                err_info = 'ReDownloading %s for %d time' % (video_zip, count)
                print(err_info)
                count += 1
        if count > 5:
            print('downloading %s failed!' % video_zip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--dataset',
        '-d',
        help='dataset name',
        default='vot-st2018',
        choices=[
            'vot-st2018',
            'vot-lt2018',
            'vot-st2019',
            'vot-lt2019',
            'vot-rgbd2019',
            'vot-rgbt2019',
            'vot-st2020',
            'vot-rgbt2020',
            'vot-st2021',
        ],
    )
    parser.add_argument(
        '--path', '-p', help='dataset saved path', default='./')
    args = parser.parse_args()
    path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), '../../../', args.path))
    if not os.path.isdir(path):
        os.makedirs(path)
    download(args.dataset, path)
