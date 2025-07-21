import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseClassifier(nn.Module, ABC):
    """Classe base abstrata para todos os classificadores"""
    
    def __init__(self, 
                 num_classes: int = 2,
                 pretrained: bool = True,
                 freeze_layers: bool = True,
                 dropout: float = 0.5):
        super(BaseClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.dropout = dropout
        
        self._build_model()
        if freeze_layers:
            self._freeze_features()
        self._initialize_weights()
    
    @abstractmethod
    def _build_model(self):
        """Constrói a arquitetura do modelo - deve ser implementado por cada subclasse"""
        pass
    
    @abstractmethod
    def _freeze_features(self):
        """Congela as camadas de feature extraction"""
        pass
    
    @abstractmethod
    def _initialize_weights(self):
        """Inicializa os pesos das camadas modificadas"""
        pass
    
    @abstractmethod
    def unfreeze_layers(self, num_layers: int = -1):
        """Descongela camadas específicas"""
        pass
    
    def get_trainable_params(self):
        """Retorna número de parâmetros treináveis"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Retorna informações do modelo"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_params()
        
        return {
            'model_name': self.__class__.__name__,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'num_classes': self.num_classes
        }