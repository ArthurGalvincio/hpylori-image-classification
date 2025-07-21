import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List, Dict, Union

from utils import get_val_transforms
from ..models import create_model


class Predictor:
    '''Classe para fazer predições em novas imagens'''
    
    def __init__(self, checkpoint_path: str, device: torch.device = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Carregar modelo
        self.model, self.config = self._load_model(checkpoint_path)
        
        # Preparar transformações
        self.transform = get_val_transforms(
            img_size=self.config['transforms']['img_size'],
            normalize_imagenet=self.config['transforms']['normalize']
        )
        
        self.class_names = self.config['data']['class_names']
    
    def _load_model(self, checkpoint_path: str):
        '''Carrega modelo do checkpoint'''
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False) 
        config = checkpoint['config']
        
        model = create_model(config) 
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def predict_image(self, image_path: Union[str, Path]) -> Dict:
        '''Faz predição para uma única imagem'''
        
        # Carregar e preprocessar imagem
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Fazer predição
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        # Preparar resultado
        result = {
            'image_path': str(image_path),
            'predicted_class': self.class_names[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                class_name: prob.item() 
                for class_name, prob in zip(self.class_names, probs[0])
            }
        }
        
        return result
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        '''Faz predição para múltiplas imagens'''
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Erro ao processar {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory_path: Union[str, Path]) -> List[Dict]:
        '''Faz predição para todas as imagens em um diretório'''
        directory_path = Path(directory_path)
        
        # Encontrar todas as imagens
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_paths = [
            p for p in directory_path.rglob('*') 
            if p.suffix.lower() in image_extensions
        ]
        
        print(f"Encontradas {len(image_paths)} imagens em {directory_path}")
        
        return self.predict_batch(image_paths)


def main():
    parser = argparse.ArgumentParser(description='Fazer predições com modelo treinado')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Caminho para checkpoint do modelo')
    parser.add_argument('--image', type=str, default=None,
                       help='Caminho para uma imagem')
    parser.add_argument('--directory', type=str, default=None,
                       help='Caminho para diretório com imagens')
    parser.add_argument('--output', type=str, default=None,
                       help='Arquivo para salvar resultados (JSON)')
    
    args = parser.parse_args()
    
    # Verificar argumentos
    if args.image is None and args.directory is None:
        parser.error("Forneça --image ou --directory")
    
    # Criar predictor
    predictor = Predictor(args.checkpoint)
    
    # Fazer predições
    if args.image:
        result = predictor.predict_image(args.image)
        results = [result]
        
        # Imprimir resultado
        print(f"\nImagem: {result['image_path']}")
        print(f"Predição: {result['predicted_class']}")
        print(f"Confiança: {result['confidence']:.2%}")
        print(f"Probabilidades:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
    
    else:  # directory
        results = predictor.predict_directory(args.directory)
        
        # Imprimir resumo
        print(f"\nResultados para {len(results)} imagens:")
        for class_name in predictor.class_names:
            count = sum(1 for r in results 
                       if 'predicted_class' in r and r['predicted_class'] == class_name)
            print(f"  {class_name}: {count}")
    
    # Salvar resultados se solicitado
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()