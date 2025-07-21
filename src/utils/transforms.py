from torchvision import transforms
from PIL import Image
import torch
import random
import numpy as np


class CustomTransform:
    '''Transformação customizada com padding'''
    
    def __init__(self, size=224):
        self.size = size
    
    def __call__(self, img):
        img.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        
        new_img = Image.new('RGB', (self.size, self.size), (0, 0, 0))
        paste_x = (self.size - img.width) // 2
        paste_y = (self.size - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img


def get_train_transforms(img_size=224, 
                        augmentation_level='medium',
                        normalize_imagenet=True):
    '''Retorna transformações para treino com diferentes níveis de augmentation'''
    
    transforms_list = [transforms.Resize((img_size + 32, img_size + 32))]
    
    if augmentation_level == 'light':
        transforms_list.extend([
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    
    elif augmentation_level == 'medium':
        transforms_list.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
        ])
    
    elif augmentation_level == 'heavy':
        transforms_list.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                 saturation=0.3, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
        ])
    
    transforms_list.append(transforms.ToTensor())
    
    if normalize_imagenet:
        transforms_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transforms_list)


def get_val_transforms(img_size=224, normalize_imagenet=True):
    '''Retorna transformações para validação/teste'''
    
    transforms_list = [
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ]
    
    if normalize_imagenet:
        transforms_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transforms_list)


class MixupTransform:
    '''Implementação de Mixup para data augmentation'''
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_images, batch_labels):
        batch_size = batch_images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        indices = torch.randperm(batch_size)
        mixed_images = lam * batch_images + (1 - lam) * batch_images[indices]
        
        labels_a, labels_b = batch_labels, batch_labels[indices]
        
        return mixed_images, labels_a, labels_b, lam