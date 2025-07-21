import sys
import os
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import numpy as np
import random
import torch.multiprocessing as mp


from ..utils import (
    IndexBasedDataset,
    criar_e_salvar_indices,
    get_train_transforms,
    get_val_transforms
)
from ..models import create_model, get_available_models
from ..training import Trainer


def set_seed(seed: int):
    '''Configura seeds para reprodutibilidade'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    '''Carrega configurações do arquivo YAML'''
    with open(config_path, 'r', encoding='utf-8') as f: 
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict):
    '''Valida se a configuração está correta'''
    model_name = config['model']['name'].lower()
    available_models = get_available_models()
    
    if model_name not in available_models:
        raise ValueError(f"Modelo '{model_name}' não está disponível. "
                         f"Modelos disponíveis: {available_models}")
    
    # Validação específica para Inception V3
    if model_name == 'inception_v3':
        if config['transforms']['img_size'] != 299:
            print("AVISO: Inception V3 requer img_size=299. Ajustando automaticamente.")
            config['transforms']['img_size'] = 299
        
        # Ajustar batch size se muito grande
        if config['dataloader']['batch_size'] > 32:
            print("AVISO: Batch size muito grande para Inception V3. Ajustando para 24.")
            config['dataloader']['batch_size'] = 24
    
    return config


def prepare_data(config: dict):
    '''Prepara datasets e dataloaders'''
    
    # Expandir caminhos
    data_path = Path(config['data']['root_path']).expanduser()
    indices_path = Path(config['data']['indices_path']).expanduser()
    
    # Criar índices se não existirem
    if not indices_path.exists():
        print("Criando índices...")
        indices = criar_e_salvar_indices(
            str(data_path),
            str(indices_path),
            train_ratio=config['splits']['train_ratio'],
            val_ratio=config['splits']['val_ratio'],
            test_ratio=config['splits']['test_ratio'],
            seed=config['splits']['seed']
        )
    
    # Preparar transformações
    train_transform = get_train_transforms(
        img_size=config['transforms']['img_size'],
        augmentation_level=config['transforms']['augmentation']['train'],
        normalize_imagenet=config['transforms']['normalize']
    )
    
    val_transform = get_val_transforms(
        img_size=config['transforms']['img_size'],
        normalize_imagenet=config['transforms']['normalize']
    )
    
    # Criar datasets
    train_dataset = IndexBasedDataset(
        root_dir=str(data_path),
        split='train',
        indices_file=str(indices_path),
        transform=train_transform
    )
    
    val_dataset = IndexBasedDataset(
        root_dir=str(data_path),
        split='val',
        indices_file=str(indices_path),
        transform=val_transform
    )
    
    test_dataset = IndexBasedDataset(
        root_dir=str(data_path),
        split='test',
        indices_file=str(indices_path),
        transform=val_transform
    )
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        persistent_workers=config['dataloader']['persistent_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        persistent_workers=config['dataloader']['persistent_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        persistent_workers=config['dataloader']['persistent_workers']
    )
    
    print(f"\nDatasets carregados:")
    print(f"Treino: {len(train_dataset)} imagens")
    print(f"Validação: {len(val_dataset)} imagens")
    print(f"Teste: {len(test_dataset)} imagens")
    print(f"Tamanho da imagem: {config['transforms']['img_size']}x{config['transforms']['img_size']}")
    print(f"Batch size: {config['dataloader']['batch_size']}")
    
    return train_loader, val_loader, test_loader


def main(config_path: str):
    '''Função principal de treinamento'''
    
    # Carregar e validar configurações
    config = load_config(config_path)
    config = validate_config(config)
    
    print(f"Configuração carregada: {config_path}")
    print(f"Modelo selecionado: {config['model']['name']}")
    print(f"Experimento: {config['experiment']['name']}")
    
    # Configurar reprodutibilidade
    if config['reproducibility']['deterministic']:
        set_seed(config['reproducibility']['seed'])
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Preparar dados
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Criar modelo
    model = create_model(config)
    
    # Criar trainer
    trainer = Trainer(model, config, device)
    
    # --- NOVO: Lógica para retomar do checkpoint ---
    start_epoch = 0 # Inicia na época 0 por padrão
    if args.resume_from: # <<< NOVO: Se o argumento --resume_from foi fornecido
        print(f"\nRetomando treino do checkpoint: {args.resume_from}")
        try:
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict']) # Carrega pesos do modelo
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Carrega estado do otimizador
            
            if trainer.scheduler and checkpoint.get('scheduler_state_dict'): # Carrega scheduler se existir
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1 # Define a época de início
            trainer.best_val_acc = checkpoint['best_val_acc'] # <<< MODIFICADO: Carrega best_val_acc correto
            trainer.best_model_path = Path(checkpoint['best_model_path']).expanduser() if checkpoint.get('best_model_path') else None # <<< NOVO: Carrega o melhor caminho
            trainer.history = checkpoint['history'] # Carrega histórico para continuar os gráficos
            print(f"Treino será retomado a partir da época {start_epoch}.")
        except Exception as e:
            print(f"Erro ao carregar o checkpoint de resume: {e}. Iniciando treino do zero.")
            start_epoch = 0 # Em caso de erro, começa do zero
    # --- FIM DA Lógica para retomar ---
    
    # Treinar
    print(f"\n{'='*60}")
    print(f"INICIANDO TREINAMENTO - {config['model']['name'].upper()}")
    print(f"{'='*60}")
    
    # >>> MODIFICADO: Passar start_epoch para o método train do Trainer <<<
    history = trainer.train(train_loader, val_loader, num_epochs=config['training']['epochs'], start_epoch=start_epoch)
    
    # Avaliar no conjunto de teste
    print(f"\n{'='*60}")
    print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print(f"{'='*60}")
    
    test_metrics = trainer.validate(test_loader)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Plotar matriz de confusão para teste
    trainer.model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = trainer.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Salvar matriz de confusão
    cm_path = trainer.results_dir / f"{config['experiment']['name']}_confusion_matrix.png"
    trainer.metrics.plot_confusion_matrix(all_labels, all_preds, save_path=str(cm_path))
    
    print(f"\n{'='*60}")
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Resultados salvos em: {trainer.results_dir}")
    print(f"Melhor modelo salvo em: {trainer.best_model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.freeze_support() 

    parser = argparse.ArgumentParser(description='Treinar modelo de classificação')
    parser.add_argument('--config', type=str, default='configs/config_base.yaml',
                        help='Caminho para arquivo de configuração')
    parser.add_argument('--list-models', action='store_true',
                        help='Lista modelos disponíveis')
    parser.add_argument('--resume_from', type=str, default=None, # <<< NOVO ARGUMENTO
                        help='Caminho para o checkpoint a partir do qual retomar o treinamento.')
    
    args = parser.parse_args()
    
    if args.list_models:
        available_models = get_available_models()
        print("Modelos disponíveis:")
        for model in available_models:
            print(f"  - {model}")
        print(f"\nPara usar um modelo, defina 'name: {available_models[0]}' na configuração.")
    else:
        main(args.config)