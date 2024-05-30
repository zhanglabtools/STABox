import requests
from tqdm import tqdm
import os
import pandas as pd
import shutil
from urllib.request import urlopen, Request
import tempfile
import ssl
from requests.packages import urllib3
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context


def download_url_to_file(url, dst, dst_save, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.
        borrow from torchvision
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # dst = os.path.expanduser(dst)
    # dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst_save)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


class SpatialDB:
    def __init__(self):
        self.url = "https://www.spatialomics.org/SpatialDB/download/"
        self.save_path = None
        self.SpatialDB_data_info_path = None
        self.session = requests.Session()
        self.file_exist()

    def file_exist(self):
        current_file_path = os.path.abspath(__file__)
        test_file_path = os.path.dirname(current_file_path)
        running_path = os.path.dirname(test_file_path)
        if not os.path.exists(os.path.join(running_path, 'dataset', 'SpatialDB_data')):
            os.makedirs(os.path.join(running_path, 'dataset', 'SpatialDB_data'))
        self.save_path = os.path.join(running_path, 'dataset', 'SpatialDB_data')
        self.SpatialDB_data_info_path = os.path.join(running_path, 'dataset', 'SpatialDB_data_info.txt')

    def get_download_data_type(self):
        data_info = pd.read_csv(self.SpatialDB_data_info_path, sep='\t')
        technique = list(data_info['Technique'].unique())
        return technique

    def get_download_data_info(self,technique):
        data_info = pd.read_csv(self.SpatialDB_data_info_path, sep='\t')
        data_type = list(data_info['Technique'].unique())
        if technique in data_type:
            technique_data_type = data_info[data_info['Technique']==technique]
            download_data_type = list(technique_data_type['Expression'].values)
        else:
            raise "Make sure your SpatialDB_data_info file is exist"

        return download_data_type

    def download(self, download_data_id):
        download_data_id= download_data_id + '.tar.gz'
        # download_url_to_file(self.url + download_data_id, self.save_path)
        download_url_to_file(self.url + download_data_id, self.save_path, self.save_path +'\\'+ download_data_id)
        # response = requests.get(self.url + download_data_id, stream=True)
        # total_size = int(response.headers.get('content-length', 0))
        # block_size = 4096
        # num_blocks = total_size // block_size + 1
        # progress_bar = tqdm(total=num_blocks, unit='blocks')
        # with open(self.save_path, 'wb') as f:
        #     for data in response.iter_content(block_size):
        #         if data:
        #             progress_bar.update()
        #             f.write(data)
        # progress_bar.close()
        # response.close()