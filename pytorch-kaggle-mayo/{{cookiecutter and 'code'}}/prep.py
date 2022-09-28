# from https://www.kaggle.com/competitions/mayo-clinic-strip-ai/discussion/338299

import os
import gc
import zipfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import cv2
import pyvips

def tile(img, sz=128, N=16):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img

def save_dataset(
    df: pd.DataFrame,
    N=16,
    max_size=20000,
    crop_size=1024,
    image_dir='../input/test',
    out_dir='test_images',
):
    if os.path.exists(out_dir):
        print(f'{out_dir} exists. skipping preprocess step...')
        return

    os.makedirs(out_dir, exist_ok=True)
    format_to_dtype = {
       'uchar': np.uint8,
       'char': np.int8,
       'ushort': np.uint16,
       'short': np.int16,
       'uint': np.uint32,
       'int': np.int32,
       'float': np.float32,
       'double': np.float64,
       'complex': np.complex64,
       'dpcomplex': np.complex128,
    }
    def vips2numpy(vi):
        return np.ndarray(
            buffer=vi.write_to_memory(),
            dtype=format_to_dtype[vi.format],
            shape=[vi.height, vi.width, vi.bands])
    
    tk0 = tqdm(enumerate(df["image_id"].values), total=len(df))
    for i, image_id in tk0:
        print(f"[{i+1}/{len(df)}] image_id: {image_id}")
        image = pyvips.Image.thumbnail(f'{image_dir}/{image_id}.tif', max_size)
        image = vips2numpy(image)
        width, height, c = image.shape
        print(f"Input width: {width} height: {height}")
        images = tile(image, sz=crop_size, N=N)
        for idx, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{out_dir}/{image_id}_{idx}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        del img, image, images; gc.collect()
