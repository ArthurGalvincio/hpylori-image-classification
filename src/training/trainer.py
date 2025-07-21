import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

from .metrics import BinaryClassificationMetrics
from ..utils.hardware_monitor import create_simple_monitor


class Trainer:
    '''Classe principal para treinamento do modelo'''
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 device: torch.device):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Configurar otimizador
        self.optimizer = self._create_optimizer()
        
        # Configurar scheduler
        self.scheduler = self._create_scheduler()
        
        # Configurar loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Métricas
        self.metrics = BinaryClassificationMetrics(
            class_names=config['data']['class_names']
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
        
        # Histórico
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'lr': []
        }
        
        # Configurar diretórios
        self.save_dir = Path(config['experiment']['save_dir']).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path(config['experiment']['results_dir']).expanduser()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # Monitor de hardware
        self.enable_monitoring = config.get('monitoring', {}).get('enabled', True)
        if self.enable_monitoring:
            monitor_interval = config.get('monitoring', {}).get('interval', 30)
            self.hardware_monitor = create_simple_monitor(log_interval=monitor_interval)
            print("Monitor de hardware ativado")
        else:
            self.hardware_monitor = None
            if self.enable_monitoring: # Este print só aparece se o monitor não está disponível, mas está habilitado no config
                print("Monitor de hardware não disponível (instale: pip install pynvml GPUtil)")
    
    def _create_optimizer(self) -> optim.Optimizer:
        '''Cria otimizador baseado na configuração'''
        opt_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        wd = self.config['training']['weight_decay']
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Otimizador {opt_name} não suportado")
    
    def _create_scheduler(self) -> Optional[object]:
        '''Cria scheduler baseado na configuração'''
        scheduler_config = self.config['training']['scheduler']
        name = scheduler_config['name'].lower()
        
        if name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                min_lr=scheduler_config['min_lr'],
            )
        elif name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config['min_lr']
            )
        elif name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        '''Treina uma época'''
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Atualizar barra de progresso
            acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        '''Valida o modelo'''
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
        # Calcular métricas
        epoch_loss = running_loss / len(val_loader)
        metrics = self.metrics.calculate(all_labels, all_preds)
        metrics['loss'] = epoch_loss
        
        return metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: Optional[int] = None,
              start_epoch: int = 0): # <<< NOVO PARÂMETRO
        '''Loop principal de treinamento'''

        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        print(f"\nIniciando treinamento por {num_epochs} épocas...")
        print(f"Modelo: {self.config['model']['name']}")
        
        # Mostrar informações do modelo se disponível
        if hasattr(self.model, 'get_trainable_params'):
            trainable_params = self.model.get_trainable_params()
            print(f"Parâmetros treináveis: {trainable_params:,}")
        else:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Parâmetros treináveis: {trainable_params:,}")
        
        # 🔍 Iniciar monitoramento de hardware
        if self.hardware_monitor:
            self.hardware_monitor.start_monitoring()
        
        training_start_time = datetime.now()
        
        # >>> MUDANÇA NO LOOP AQUI <<<
        for epoch in range(start_epoch, num_epochs): # <<< AGORA COMEÇA DE start_epoch
            print(f"\nÉpoca {epoch+1}/{num_epochs}") # epoch+1 para exibir o número da época corretamente
            print("-" * 50)
            
            # Treinar
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validar
            val_metrics = self.validate(val_loader)
            
            # Atualizar histórico
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print métricas
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Val F1: {val_metrics['f1_score']:.4f} | Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Salvar melhor modelo
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"Novo melhor modelo salvo! Acurácia: {self.best_val_acc:.2f}%")
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping ativado na época {epoch+1}")
                break
            
            # Salvar checkpoint periódico
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        training_end_time = datetime.now()
        training_duration = training_end_time - training_start_time
        
        # 🔍 Parar monitoramento e mostrar resumo
        if self.hardware_monitor:
            self.hardware_monitor.stop_monitoring()
            
            # Mostrar resumo do hardware
            self.hardware_monitor.print_summary()
            
            # Salvar log detalhado
            hardware_log_path = self.results_dir / f"{self.config['experiment']['name']}_hardware_log.json"
            self.hardware_monitor.save_detailed_log(str(hardware_log_path))
        
        # Salvar resultados finais
        self.save_training_results()
        self.plot_training_history()
        
        print("\n" + "="*60)
        print("TREINAMENTO CONCLUÍDO!")
        print("="*60)
        print(f"  Tempo total: {training_duration}")
        print(f" Melhor acurácia de validação: {self.best_val_acc:.2f}%")
        print(f" Modelo salvo em: {self.best_model_path}")
        print("="*60)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        '''Salva checkpoint do modelo'''
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        if is_best:
            filename = self.save_dir / 'best_model.pth'
            self.best_model_path = filename
        else:
            filename = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, filename)
        if not is_best:  # Não printar para checkpoints periódicos
            print(f"Checkpoint salvo: {filename}")
    
    def save_training_results(self):
        '''Salva resultados do treinamento'''
        results = {
            'config': self.config,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_model_path': str(self.best_model_path) if self.best_model_path else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results_file = self.results_dir / f"{self.config['experiment']['name']}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Resultados salvos em: {results_file}")
    
    def plot_training_history(self):
        '''Plota gráficos do histórico de treinamento'''
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1, Precision, Recall
        axes[1, 0].plot(self.history['val_f1'], label='F1 Score', linewidth=2)
        axes[1, 0].plot(self.history['val_precision'], label='Precision', linewidth=2)
        axes[1, 0].plot(self.history['val_recall'], label='Recall', linewidth=2)
        axes[1, 0].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['lr'], linewidth=2, color='orange')
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.results_dir / f"{self.config['experiment']['name']}_history.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Gráficos salvos em: {plot_file}")


class EarlyStopping:
    '''Early stopping para evitar overfitting'''
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop