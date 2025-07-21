import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import os
import pandas as pd
from typing import Optional, Callable, Tuple, Dict, List


class CustomImageDataset(Dataset):
    '''Dataset customizado para imagens com suporte a CSV'''
    
    def __init__(self, 
                 root_dir: str,
                 annotations_file: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        if annotations_file:
            self.annotations = pd.read_csv(annotations_file)
        else:
            self.annotations = self._scan_directory()
        
        self.classes = sorted(self.annotations['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def _scan_directory(self) -> pd.DataFrame:
        data = []
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data.append({
                            'filename': os.path.join(class_name, img_name),
                            'label': class_name
                        })
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['filename'])
        label = self.annotations.iloc[idx]['label']
        
        image = Image.open(img_path).convert('RGB')
        label_idx = self.class_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label_idx = self.target_transform(label_idx)
        
        return image, label_idx


class AdvancedImageDataset(Dataset):
    '''Dataset avançado com cache e metadados'''
    
    def __init__(self, 
                 data_config: dict,
                 transform: Optional[Callable] = None,
                 cache_images: bool = False):
        
        self.transform = transform
        self.cache_images = cache_images
        self.cached_data = {}
        
        self.root_dir = data_config['root_dir']
        self.image_list = data_config['images']
        self.labels = data_config['labels']
        self.metadata = data_config.get('metadata', {})
        
        if self.cache_images:
            self._cache_all_images()
    
    def _cache_all_images(self):
        print("Carregando imagens na memória...")
        for idx in range(len(self)):
            img_path = os.path.join(self.root_dir, self.image_list[idx])
            self.cached_data[idx] = Image.open(img_path).convert('RGB')
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        if self.cache_images:
            image = self.cached_data[idx]
        else:
            img_path = os.path.join(self.root_dir, self.image_list[idx])
            image = Image.open(img_path).convert('RGB')
        
        output_data = {
            'label': self.labels[idx],
            'filename': self.image_list[idx],
            'index': idx
        }
        
        if idx in self.metadata:
            output_data.update(self.metadata[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, output_data


class IndexBasedDataset(Dataset):
    '''Dataset que usa arquivos de índices para splits'''
    
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 indices_file: str = None,
                 transform=None):
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.full_dataset = datasets.ImageFolder(root=root_dir)
        
        if indices_file and os.path.exists(indices_file):
            indices_data = torch.load(indices_file)
            self.indices = indices_data[split]
            self.classes = indices_data['classes']
            self.class_to_idx = indices_data['class_to_idx']
        else:
            self.indices = list(range(len(self.full_dataset)))
            self.classes = self.full_dataset.classes
            self.class_to_idx = self.full_dataset.class_to_idx
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.full_dataset[real_idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label