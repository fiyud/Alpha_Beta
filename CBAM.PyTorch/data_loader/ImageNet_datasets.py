import torch
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image
import random
import os
import cv2
import random
import urllib.request
import tarfile
import zipfile
import shutil
from tqdm import tqdm

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        raise(RuntimeError('No Module named accimage'))
    else:
        return pil_loader(path)

def download_url(url, destination, progress=True):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    if os.path.exists(destination):
        print(f"File already exists at {destination}")
        return
    
    print(f"Downloading {url} to {destination}")
    
    def _progress(count, block_size, total_size):
        if total_size > 0 and progress:
            percent = count * block_size * 100 // total_size
            progress_bar = "[" + "#" * (percent // 5) + " " * (20 - percent // 5) + "]"
            print(f"\r{progress_bar} {percent}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=_progress if progress else None)
        print("\nDownload complete!")
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise e

def extract_archive(archive_path, extract_path):
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"Extracting {archive_path} to {extract_path}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting'):
                try:
                    zip_ref.extract(member, extract_path)
                except zipfile.error:
                    pass
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            for member in tqdm(tar.getmembers(), desc='Extracting'):
                try:
                    tar.extract(member, extract_path)
                except tarfile.error:
                    pass
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print("Extraction complete!")

def download_imagenet(root_dir, dataset_type="tiny-imagenet"):
    os.makedirs(root_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(root_dir, 't256')) and os.path.exists(os.path.join(root_dir, 'v256')):
        print(f"Dataset already exists at {root_dir}")
        return True
    
    if dataset_type == "tiny-imagenet":
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        archive_path = os.path.join(root_dir, "tiny-imagenet-200.zip")
        
        try:
            download_url(url, archive_path)
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return False
        
        try:
            extract_archive(archive_path, root_dir)
        except Exception as e:
            print(f"Error extracting dataset: {str(e)}")
            return False
        
        try:
            t256_dir = os.path.join(root_dir, 't256')
            v256_dir = os.path.join(root_dir, 'v256')
            os.makedirs(t256_dir, exist_ok=True)
            os.makedirs(v256_dir, exist_ok=True)
            
            train_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
            print("Processing training data...")
            for class_dir in tqdm(os.listdir(train_dir)):
                class_path = os.path.join(train_dir, class_dir)
                if os.path.isdir(class_path):
                    os.makedirs(os.path.join(t256_dir, class_dir), exist_ok=True)
                    images_dir = os.path.join(class_path, 'images')
                    if os.path.exists(images_dir):
                        for img in os.listdir(images_dir):
                            if img.endswith(('.JPEG', '.jpg', '.png')):
                                src = os.path.join(images_dir, img)
                                dst = os.path.join(t256_dir, class_dir, img)
                                shutil.copy2(src, dst)
            
            val_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val')
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            print("Processing validation data...")
            
            val_img_to_class = {}
            if os.path.exists(val_annotations_file):
                with open(val_annotations_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name, class_id = parts[0], parts[1]
                            val_img_to_class[img_name] = class_id
            
            val_images_dir = os.path.join(val_dir, 'images')
            if os.path.exists(val_images_dir):
                for img in tqdm(os.listdir(val_images_dir)):
                    if img in val_img_to_class:
                        class_id = val_img_to_class[img]
                        os.makedirs(os.path.join(v256_dir, class_id), exist_ok=True)
                        src = os.path.join(val_images_dir, img)
                        dst = os.path.join(v256_dir, class_id, img)
                        shutil.copy2(src, dst)
            
            classes = sorted(os.listdir(t256_dir))
            class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
            
            with open(os.path.join(root_dir, 'train.txt'), 'w') as f:
                for class_name in tqdm(classes, desc='Creating train.txt'):
                    class_dir = os.path.join(t256_dir, class_name)
                    if os.path.isdir(class_dir):
                        for img in os.listdir(class_dir):
                            if img.endswith(('.JPEG', '.jpg', '.png')):
                                f.write(f"{os.path.join(class_name, img)}\t{class_to_idx[class_name]}\n")
            
            # Create val.txt
            with open(os.path.join(root_dir, 'val.txt'), 'w') as f:
                for class_name in tqdm(classes, desc='Creating val.txt'):
                    class_dir = os.path.join(v256_dir, class_name)
                    if os.path.isdir(class_dir):
                        for img in os.listdir(class_dir):
                            if img.endswith(('.JPEG', '.jpg', '.png')):
                                f.write(f"{os.path.join(class_name, img)}\t{class_to_idx[class_name]}\n")
            
            os.remove(archive_path)
            print(f"Dataset successfully prepared at {root_dir}")
            return True
            
        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")
            return False
    else:
        print(f"Unsupported dataset type: {dataset_type}")
        return False

class ImageNetData(data.Dataset):
    def __init__(self, img_root, img_file, is_training=False, transform=None, target_transform=None, loader=default_loader, download=False):
        self.root = img_root
        
        if download:
            download_imagenet(img_root)
            
        if not os.path.exists(os.path.join(img_root, 't256')) or not os.path.exists(os.path.join(img_root, 'v256')):
            raise RuntimeError(f"Dataset not found at {img_root}. Use download=True to download it.")
            
        if not os.path.exists(img_file):
            raise RuntimeError(f"Index file not found at {img_file}.")

        self.imgs = []
        with open(img_file, 'r', encoding='utf-8') as fd:
            for i, _line in enumerate(fd.readlines()):
                infos = _line.replace('\n', '').split('\t')
                if 2 != len(infos):
                    continue
                if is_training:
                    real_path = os.path.join(self.root, 't256', infos[0])
                else:
                    real_path = os.path.join(self.root, 'v256', infos[0])
                class_id  = int(infos[-1])
                self.imgs.append((real_path, class_id))
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, class_id = self.imgs[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            class_id = torch.LongTensor([class_id])

        return img, class_id
    
    def __len__(self):
        return len(self.imgs)