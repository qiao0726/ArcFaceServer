import minio
import base64
import os
from io import BytesIO
from PIL import Image
from urllib import request as re
from urllib.parse import quote
from urllib.error import HTTPError

# Decode base64 to a PIL image
def base64_to_image(base64_str):  # 用 b.show()可以展示
    image = base64.b64decode(base64_str, altchars=None, validate=False)
    image = BytesIO(image)
    image = Image.open(image)
    return image.convert('RGB')

def encode_chinese_url(url:str):
    encoded_url = str()
    for uchar in url:
        if uchar >=u'\u2E80' and uchar <=u'\uFE4F': # If uchar is a chinese character
            uchar = quote(uchar)
        encoded_url += uchar
    return encoded_url
        

MINIO_CONFIG = {
    'endpoint': '192.168.1.133:9000',
    'access_key': 'minioadmin',
    'secret_key': 'minioadmin',
    'secure': False
}

client = minio.Minio(**MINIO_CONFIG)
def pull_img_from_minio(minio_url:str, local_save_path:str):
    """Download an image from minio and save it to the local file system.

    Args:
        minio_url (str): Load from
        local_save_path (str): Save to
    """
    # save_dir = local_save_path.split('/')[0:-2]
    if minio_url is None:
        return 0
    # If url contains chinese characters
    minio_url = encode_chinese_url(minio_url)
    
    # If url doesn't start with 'http'
    if minio_url[0:4] != 'http':
        minio_url = 'http://' + minio_url
    try:
        re.urlretrieve(minio_url, local_save_path)
    except HTTPError:
        return 0
    return 1

def upload2minio(upload_file_path:str, bucket_name, file_name):
    client.fput_object(bucket_name=bucket_name, object_name=file_name,
                       file_path=upload_file_path,
                       # content_type='application/zip'
                       content_type='application/picture'
                       )
    