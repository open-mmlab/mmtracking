# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import multiprocessing
import os
import os.path as osp
import re
import socket
from urllib import error, request

from tqdm import tqdm

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def download_url(url_savedir_tuple):
    url = url_savedir_tuple[0]
    saved_dir = url_savedir_tuple[1]
    video_zip = osp.basename(url)
    if not osp.isdir(saved_dir):
        os.makedirs(saved_dir, exist_ok=True)
    saved_file = osp.join(saved_dir, video_zip)

    try:
        request.urlretrieve(url, saved_file)
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
                err_info = 'Reloading %s for %d time' % (video_zip, count)
                print(err_info)
                count += 1
        if count > 5:
            print('downloading %s failed!' % video_zip)


def parse_url(homepage, href=None):
    html = request.urlopen(homepage + 'datasets.html').read().decode('utf-8')
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, features='html.parser')
    else:
        raise ImportError(
            "Please install beautifulsoup4 by 'pip install beautifulsoup4'")

    tags = soup.find_all('a', href=href)
    for tag in tags:
        yield str(tag.get('href')).strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download OTB100 dataset')
    parser.add_argument('-o', '--out', help='directory to save download file')
    parser.add_argument('-p', type=int, default=1, help='multiprocess')
    args = parser.parse_args()

    otb100_url = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/'
    href = re.compile(r'.zip$')

    all_params_tuple = [(otb100_url + video_name, args.out)
                        for video_name in parse_url(otb100_url, href)]
    with multiprocessing.Pool(processes=args.p) as pool:
        res = list(
            tqdm(
                pool.imap(download_url, all_params_tuple),
                total=len(all_params_tuple),
                desc='Downloading OTB100'))
