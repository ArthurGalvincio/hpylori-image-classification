from .vgg_classifier import VGGClassifier
from .resnet_classifier import ResNetClassifier
from .inception_classifier import InceptionClassifier
from .base_classifier import BaseClassifier

from typing import Dict
import torch.nn as nn


# Registry de modelos disponíveis
MODEL_REGISTRY = {
    'vgg16': VGGClassifier,
    'resnet50': ResNetClassifier,
    'inception_v3': InceptionClassifier,
}


def create_model(config: Dict) -> nn.Module:
    """
    Factory function para criar modelos baseado na configuração
    
    Args:
        config: Dicionário com configurações do modelo
    
    Returns:
        Instância do modelo configurado
    """
    model_name = config['model']['name'].lower()
    
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Modelo '{model_name}' não disponível. "
                        f"Modelos disponíveis: {available_models}")
    
    # Extrair parâmetros do modelo
    model_params = {
        'num_classes': config['data']['num_classes'],
        'pretrained': config['model']['pretrained'],
        'freeze_layers': config['model']['freeze_layers'],
        'dropout': config['model']['dropout']
    }
    
    # Criar instância do modelo
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(**model_params)
    
    print(f"Modelo criado: {model_name}")
    print(f"Informações do modelo:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        if 'params' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    return model


def get_available_models():
    """Retorna lista de modelos disponíveis"""
    return list(MODEL_REGISTRY.keys())


def register_model(name: str, model_class):
    """
    Registra um novo modelo no registry
    
    Args:
        name: Nome do modelo
        model_class: Classe do modelo
    """
    MODEL_REGISTRY[name] = model_class
    print(f"Modelo '{name}' registrado com sucesso!")


# Exportar tudo que for necessário
__all__ = [
    'create_model',
    'get_available_models', 
    'register_model',
    'VGGClassifier',
    'ResNetClassifier', 
    'InceptionClassifier',
    'BaseClassifier'
]