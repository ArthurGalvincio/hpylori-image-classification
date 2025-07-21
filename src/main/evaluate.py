import datetime
import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json

from torchvision import datasets
from ..utils import IndexBasedDataset, get_val_transforms
from ..models import create_model
from ..training.metrics import BinaryClassificationMetrics
from ..utils.inference_monitor import create_inference_monitor



def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Carrega modelo do checkpoint usando o sistema flexível
    Detecta automaticamente o tipo de modelo
    """
    print(f" Carregando checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    print(f" Modelo detectado: {config['model']['name']}")
    print(f" Classes: {config['data']['class_names']}")
    
    # Criar modelo usando o factory
    model = create_model(config)
    
    # Carregar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def evaluate_model_with_monitoring(model, test_loader, config, device, enable_monitoring=True):
    """Avalia modelo com monitoramento de performance de inferência"""
    
    # Inicializar métricas
    metrics = BinaryClassificationMetrics(
        class_names=config['data']['class_names']
    )
    
    # Inicializar monitor de inferência
    inference_monitor = None
    if enable_monitoring:
        inference_monitor = create_inference_monitor(log_interval=5)
        total_images = len(test_loader.dataset)
        inference_monitor.start_inference(total_images)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=' Avaliando')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            # Timing do batch
            batch_start_time = time.time()
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Sincronizar GPU para timing preciso
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            # Log do batch no monitor
            if inference_monitor:
                inference_monitor.log_batch(len(inputs), batch_time)
            
            # Coletar predições
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Atualizar barra de progresso com throughput
            if batch_time > 0:
                throughput = len(inputs) / batch_time
                pbar.set_postfix({
                    'throughput': f'{throughput:.1f} img/s',
                    'batch_time': f'{batch_time*1000:.1f}ms'
                })
    
    # Parar monitoramento
    if inference_monitor:
        inference_monitor.stop_inference()
    
    # Calcular métricas de acurácia
    test_metrics = metrics.calculate(all_labels, all_preds)
    
    return test_metrics, all_labels, all_preds, all_probs, inference_monitor


def save_evaluation_results(config, test_metrics, inference_monitor, model_info, output_path=None):
    """Salva resultados da avaliação"""
    if output_path:
        output_file_path = Path(output_path).expanduser()
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    else: 
        results_dir = Path(config['experiment']['results_dir']).expanduser()
        results_dir.mkdir(parents=True, exist_ok=True)
        experiment_name = config['experiment']['name']
        output_file_path = results_dir / f"{experiment_name}_evaluation_inference.json" 

    data_to_save = {} 
    if inference_monitor:
        data_to_save['summary'] = inference_monitor.get_inference_summary()
        data_to_save['batch_stats'] = inference_monitor.batch_stats 
        data_to_save['raw_times'] = inference_monitor.raw_batch_times
        data_to_save['batch_sizes'] = inference_monitor.raw_batch_sizes
        data_to_save['monitoring_config'] = inference_monitor.monitoring_config

    data_to_save['test_metrics'] = test_metrics # 
    data_to_save['model_info'] = model_info 

    data_to_save['timestamp'] = datetime.now().isoformat()

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=2)

    print(f" Resultados de avaliação salvos em: {output_file_path}")


def generate_evaluation_plots(config, all_labels, all_preds, all_probs, metrics, output_base_name=None):
    """Gera plots da avaliação (Matriz de Confusão, Curva ROC)"""
    results_dir = Path(config['experiment']['results_dir']).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Usa o nome base para os plots, ou o nome do experimento padrão
    plot_base_name = output_base_name if output_base_name else config['experiment']['name'] 

    # Matriz de confusão
    cm_path = results_dir / f"{plot_base_name}_evaluation_confusion_matrix.png" 
    metrics.plot_confusion_matrix(all_labels, all_preds, save_path=str(cm_path))
    print(f" Matriz de confusão salva: {cm_path}")

    # Curva ROC (apenas para classificação binária)
    if config['data']['num_classes'] == 2:
        roc_path = results_dir / f"{plot_base_name}_evaluation_roc_curve.png" 
        metrics.plot_roc_curve(all_labels, np.array(all_probs), save_path=str(roc_path))
        print(f" Curva ROC salva: {roc_path}")


def main():
    parser = argparse.ArgumentParser(description='Avaliar modelo treinado com monitoramento de inferência')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Caminho para checkpoint do modelo (.pth)')
    parser.add_argument('--data', type=str, default=None,
                       help='Caminho para dados (sobrescreve config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size para avaliação (sobrescreve config)')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Desabilitar monitoramento de inferência')
    parser.add_argument('--estimate-hardware', type=int, default=None,
                       help='Estimar hardware para N imagens')
    parser.add_argument('--target-time', type=float, default=None,
                       help='Tempo alvo em minutos para estimativa')
    parser.add_argument('--output', type=str, default=None, 
                        help='Caminho completo para salvar o arquivo JSON de resultados da avaliação. Ex: results/vgg16/vgg16_deepHP_evaluation.json')
    parser.add_argument('--device', type=str, default=None, 
                        help='Dispositivo para usar para avaliação (ex: cpu, cuda:0). Sobrescreve detecção automática.')
    
    args = parser.parse_args()
    
    # Validar argumentos
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f" Checkpoint não encontrado: {checkpoint_path}")
        return
    
    # Configurar dispositivo (GPU se disponível, senão CPU)
    if args.device: # Se o argumento --device foi fornecido
        device = torch.device(args.device)
    else: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Dispositivo: {device}")
    if device.type == 'cuda' and torch.cuda.is_available(): # Verifica se é GPU e se está disponível
        print(f" GPU: {torch.cuda.get_device_name()}")
    
    # Carregar modelo
    model, config = load_model_from_checkpoint(str(checkpoint_path), device)
    
    # Configurar dados
    data_path = args.data if args.data else config['data']['root_path']
    data_path = Path(data_path).expanduser()
    
    print(f" Dataset: {data_path}")
    
    # Preparar transformações
    test_transform = get_val_transforms(
        img_size=config['transforms']['img_size'],
        normalize_imagenet=config['transforms']['normalize']
    )
    
    # Dataset de teste
    if args.data: 
        print(f" Usando ImageFolder para dataset externo: {data_path}")
        test_dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=test_transform
        )
        config['data']['class_names'] = test_dataset.classes 
        print(f" Classes detectadas no dataset externo: {test_dataset.classes}")
    else: 
        print(f" Usando IndexBasedDataset com indices: {config['data']['indices_path']}")
        indices_path = Path(config['data']['indices_path']).expanduser()
        test_dataset = IndexBasedDataset(
            root_dir=str(data_path),
            split='test', # Assume que a avaliação é no split 'test'
            indices_file=str(indices_path),
            transform=test_transform
        )
    
    # DataLoader com batch_size configurável
    batch_size = args.batch_size if args.batch_size else config['dataloader']['batch_size']
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory']
    )
    
    print(f" Dataset de teste: {len(test_dataset):,} imagens")
    print(f" Batch size: {batch_size}")
    print(f" Total de batches: {len(test_loader)}")
    
    # Informações do modelo para log
    model_info = {
        'model_name': config['model']['name'],
        'num_classes': config['data']['num_classes'],
        'img_size': config['transforms']['img_size'],
        'batch_size': batch_size,
        'device': str(device),
        'checkpoint_path': str(checkpoint_path)
    }
    
    if hasattr(model, 'get_model_info'):
        model_info.update(model.get_model_info())
    
    # Avaliar modelo
    print("\n" + "="*60)
    print(" INICIANDO AVALIAÇÃO COM MONITORAMENTO")
    print("="*60)
    
    enable_monitoring = not args.no_monitoring
    if not enable_monitoring and not args.no_monitoring:
        print("  Monitor de inferência não disponível (instale: pip install psutil pynvml GPUtil)")
    
    test_metrics, all_labels, all_preds, all_probs, inference_monitor = evaluate_model_with_monitoring(
        model, test_loader, config, device, enable_monitoring
    )
    
    # Mostrar resultados de acurácia
    print("\n" + "="*60)
    print(" RESULTADOS DE ACURÁCIA")
    print("="*60)
    print(f" Acurácia: {test_metrics['accuracy']:.2f}%")
    print(f" F1 Score: {test_metrics['f1_score']:.4f}")
    print(f" Precisão: {test_metrics['precision']:.4f}")
    print(f" Recall: {test_metrics['recall']:.4f}")
    
    # Mostrar relatório detalhado
    print(f"\n Relatório de Classificação:")
    metrics_obj = BinaryClassificationMetrics(class_names=config['data']['class_names'])
    metrics_obj.print_classification_report(all_labels, all_preds)
    
    # Mostrar resultados de performance
    if inference_monitor:
        inference_monitor.print_inference_summary()
    
    # Gerar plots
    plot_base_name = Path(args.output).stem if args.output else config['experiment']['name'] 
    generate_evaluation_plots(config, all_labels, all_preds, all_probs, metrics_obj, output_base_name=plot_base_name) 

    # Salvar resultados
    save_evaluation_results(config, test_metrics, inference_monitor, model_info, output_path=args.output) 
    
    print("\n" + "="*60)
    print(" AVALIAÇÃO CONCLUÍDA!")
    print("="*60)
    
    # Resumo final para cotação de hardware
    if inference_monitor:
        summary = inference_monitor.get_inference_summary()
        if summary:
            throughput = summary['throughput']['images_per_second']
            time_per_image = summary['throughput']['time_per_image_ms']
            
            print(f" RESUMO PARA COTAÇÃO:")
            print(f"   • Throughput: {throughput:.1f} imagens/segundo")
            print(f"   • Tempo por imagem: {time_per_image:.2f} ms")
            print(f"   • Modelo: {model_info['model_name']}")
            print(f"   • Hardware: {device}")

if __name__ == "__main__":
    main()