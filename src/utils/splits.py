import torch
from torch.utils.data import random_split, Subset
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
from collections import defaultdict
import pandas as pd
from typing import Tuple, List


def dividir_dataset_simples(dataset_path: str, 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          transform=None,
                          seed: int = 42) -> Tuple:
    '''Divisão aleatória simples usando random_split'''
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset, test_dataset


def dividir_dataset_estratificado(dataset_path: str,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15,
                                transform=None,
                                seed: int = 42) -> Tuple:
    '''Divisão estratificada mantendo proporção de classes'''
    
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    targets = np.array([s[1] for s in full_dataset.samples])
    
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=test_ratio,
        stratify=targets,
        random_state=seed
    )
    
    train_size_adjusted = train_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1-train_size_adjusted,
        stratify=targets[train_val_idx],
        random_state=seed
    )
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    return train_dataset, val_dataset, test_dataset


def criar_estrutura_dividida(source_path: str,
                           dest_path: str,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           seed: int = 42,
                           copy_files: bool = True) -> str:
    '''Cria estrutura física de pastas train/val/test'''
    
    np.random.seed(seed)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_path, split), exist_ok=True)
    
    for class_name in os.listdir(source_path):
        class_path = os.path.join(source_path, class_name)
        
        if not os.path.isdir(class_path):
            continue
            
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dest_path, split, class_name), exist_ok=True)
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        np.random.shuffle(images)
        
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        file_operation = shutil.copy2 if copy_files else shutil.move
        
        for img_set, split in [(train_images, 'train'), 
                               (val_images, 'val'), 
                               (test_images, 'test')]:
            for img in img_set:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_path, split, class_name, img)
                file_operation(src, dst)
    
    return dest_path


def criar_e_salvar_indices(dataset_path: str,
                         indices_path: str,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         seed: int = 42) -> dict:
    '''Cria e salva arquivo com índices dos splits'''
    
    full_dataset = datasets.ImageFolder(root=dataset_path)
    targets = np.array([s[1] for s in full_dataset.samples])
    
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=test_ratio,
        stratify=targets,
        random_state=seed
    )
    
    train_size_adjusted = train_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1-train_size_adjusted,
        stratify=targets[train_val_idx],
        random_state=seed
    )
    
    indices_data = {
        'train': train_idx.tolist(),
        'val': val_idx.tolist(),
        'test': test_idx.tolist(),
        'classes': full_dataset.classes,
        'class_to_idx': full_dataset.class_to_idx
    }
    
    torch.save(indices_data, indices_path)
    return indices_data


def verificar_balanceamento(train_dataset, val_dataset, test_dataset, 
                          dataset_classes: List[str]) -> pd.DataFrame:
    '''Verifica distribuição de classes nos splits'''
    
    def contar_classes(dataset, classes):
        counts = defaultdict(int)
        for _, label in dataset:
            counts[classes[label]] += 1
        return counts
    
    train_counts = contar_classes(train_dataset, dataset_classes)
    val_counts = contar_classes(val_dataset, dataset_classes)
    test_counts = contar_classes(test_dataset, dataset_classes)
    
    df = pd.DataFrame({
        'Treino': train_counts,
        'Validação': val_counts,
        'Teste': test_counts
    }).fillna(0).astype(int)
    
    df['Total'] = df.sum(axis=1)
    df['Treino %'] = (df['Treino'] / df['Total'] * 100).round(1)
    df['Val %'] = (df['Validação'] / df['Total'] * 100).round(1)
    df['Teste %'] = (df['Teste'] / df['Total'] * 100).round(1)
    
    return df