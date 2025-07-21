import torch
import torch.nn as nn
import torchvision.models as models
from .base_classifier import BaseClassifier


class InceptionClassifier(BaseClassifier):
    """Inception V3 adaptado para classificação"""
    
    def __init__(self, **kwargs):
        super(InceptionClassifier, self).__init__(**kwargs)
    
    def _build_model(self):
        """Constrói o modelo Inception V3"""
        # Carregar Inception V3 pré-treinado
        if self.pretrained:
            self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.inception_v3(weights=None)
        
        # Desabilitar saída auxiliar durante inferência
        self.backbone.aux_logits = False
        
        # Obter número de features da última camada
        num_features = self.backbone.fc.in_features
        
        # Substituir a camada final
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, self.num_classes)
        )
    
    def _freeze_features(self):
        """Congela as camadas de features (tudo exceto fc)"""
        for name, param in self.backbone.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
    
    def _initialize_weights(self):
        """Inicializa pesos das camadas do classificador"""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def unfreeze_layers(self, num_layers: int = -1):
        """Descongela camadas específicas"""
        if num_layers == -1:
            # Descongela todas as camadas
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Para Inception, descongelar por blocos (Mixed)
            # Mixed_7c, Mixed_7b, Mixed_7a, etc.
            all_layer_names = []
            for name, _ in self.backbone.named_parameters():
                layer_base = name.split('.')[0]
                if layer_base not in all_layer_names and 'Mixed' in layer_base:
                    all_layer_names.append(layer_base)
            
            # Ordenar em ordem reversa (últimas camadas primeiro)
            all_layer_names.sort(reverse=True)
            
            # Descongelar os últimos num_layers blocos
            layers_to_unfreeze = all_layer_names[:min(num_layers, len(all_layer_names))]
            
            for name, param in self.backbone.named_parameters():
                for layer_name in layers_to_unfreeze:
                    if name.startswith(layer_name):
                        param.requires_grad = True

        for module in self.backbone.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
    
    def forward(self, x):
        # Durante treinamento, o Inception pode retornar saídas auxiliares
        if self.training and hasattr(self.backbone, 'aux_logits') and self.backbone.aux_logits:
            output, aux_output = self.backbone(x)
            return output
        else:
            return self.backbone(x)