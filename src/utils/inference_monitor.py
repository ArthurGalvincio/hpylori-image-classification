import time
import torch
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from .hardware_monitor import HardwareMonitor


class InferenceMonitor:
    """Monitor específico para medir performance de inferência"""
    
    def __init__(self, log_interval: int = 5):
        """
        Args:
            log_interval: Intervalo em segundos para coleta de dados
        """
        self.log_interval = log_interval
        
        self.hardware_monitor = HardwareMonitor(log_interval=log_interval)
        
        # Métricas de inferência
        self.inference_times = []
        self.batch_sizes = []
        self.total_images = 0
        self.start_time = None
        self.end_time = None
        
        # Estatísticas por batch
        self.batch_stats = []
    
    def start_inference(self, total_images: int):
        """Inicia monitoramento de inferência"""
        print(" Iniciando monitoramento de inferência...")
        
        self.total_images = total_images
        self.start_time = time.time()
        
        # Resetar dados
        self.inference_times = []
        self.batch_sizes = []
        self.batch_stats = []
        
        # Iniciar monitor de hardware
        if self.hardware_monitor:
            self.hardware_monitor.start_monitoring()
    
    def log_batch(self, batch_size: int, batch_time: float):
        """Log de performance de um batch"""
        self.inference_times.append(batch_time)
        self.batch_sizes.append(batch_size)
        
        # Calcular métricas do batch
        images_per_second = batch_size / batch_time if batch_time > 0 else 0
        time_per_image = batch_time / batch_size if batch_size > 0 else 0
        
        self.batch_stats.append({
            'batch_size': batch_size,
            'batch_time': batch_time,
            'images_per_second': images_per_second,
            'time_per_image': time_per_image
        })
    
    def stop_inference(self):
        """Para monitoramento de inferência"""
        print("  Finalizando monitoramento de inferência...")
        
        self.end_time = time.time()
        
        # Parar monitor de hardware
        if self.hardware_monitor:
            self.hardware_monitor.stop_monitoring()
    
    def get_inference_summary(self) -> Dict:
        """Retorna resumo das métricas de inferência"""
        if not self.inference_times or not self.start_time or not self.end_time:
            return {}
        
        # Tempo total
        total_time = self.end_time - self.start_time
        
        # Estatísticas de tempo
        avg_batch_time = sum(self.inference_times) / len(self.inference_times)
        total_inference_time = sum(self.inference_times)
        
        # Throughput (imagens por segundo)
        avg_throughput = self.total_images / total_time if total_time > 0 else 0
        
        # Tempo por imagem
        avg_time_per_image = total_time / self.total_images if self.total_images > 0 else 0
        
        # Estatísticas dos batches
        batch_sizes = self.batch_sizes
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
        
        summary = {
            'timing': {
                'total_time_seconds': total_time,
                'total_time_formatted': str(timedelta(seconds=int(total_time))),
                'inference_time_seconds': total_inference_time,
                'overhead_time_seconds': total_time - total_inference_time,
                'overhead_percentage': ((total_time - total_inference_time) / total_time * 100) if total_time > 0 else 0
            },
            'throughput': {
                'images_per_second': avg_throughput,
                'batches_per_second': len(self.inference_times) / total_time if total_time > 0 else 0,
                'time_per_image_ms': avg_time_per_image * 1000,
                'time_per_batch_ms': avg_batch_time * 1000
            },
            'dataset': {
                'total_images': self.total_images,
                'total_batches': len(self.inference_times),
                'avg_batch_size': avg_batch_size
            },
            'performance_stats': {
                'min_batch_time': min(self.inference_times) if self.inference_times else 0,
                'max_batch_time': max(self.inference_times) if self.inference_times else 0,
                'avg_batch_time': avg_batch_time
            }
        }
        
        # Adicionar dados de hardware se disponível
        if self.hardware_monitor:
            hardware_summary = self.hardware_monitor.get_summary()
            summary['hardware'] = hardware_summary
        
        return summary
    
    def print_inference_summary(self):
        """Imprime resumo formatado da inferência"""
        summary = self.get_inference_summary()
        
        if not summary:
            print(" Nenhum dado de inferência disponível")
            return
        
        print("\n" + "="*60)
        print(" RESUMO DE PERFORMANCE DE INFERÊNCIA")
        print("="*60)
        
        # Dados do dataset
        dataset = summary['dataset']
        print(f" Dataset:")
        print(f"   Total de imagens: {dataset['total_images']:,}")
        print(f"   Total de batches: {dataset['total_batches']:,}")
        print(f"   Batch size médio: {dataset['avg_batch_size']:.1f}")
        
        # Timing
        timing = summary['timing']
        print(f"\n  Tempo:")
        print(f"   Tempo total: {timing['total_time_formatted']}")
        print(f"   Tempo de inferência: {timing['inference_time_seconds']:.2f}s")
        print(f"   Overhead: {timing['overhead_percentage']:.1f}%")
        
        # Throughput
        throughput = summary['throughput']
        print(f"\n Performance:")
        print(f"   Throughput: {throughput['images_per_second']:.1f} imagens/segundo")
        print(f"   Tempo por imagem: {throughput['time_per_image_ms']:.2f} ms")
        print(f"   Tempo por batch: {throughput['time_per_batch_ms']:.2f} ms")
        
        # Hardware (se disponível)
        if 'hardware' in summary and summary['hardware']:
            hw = summary['hardware']
            
            if hw.get('gpu', {}).get('available', False):
                gpu = hw['gpu']
                print(f"\n GPU:")
                print(f"   Uso médio: {gpu['usage']['avg']:.1f}%")
                print(f"   VRAM média: {gpu['memory']['avg']:.1f}%")
            
            system = hw.get('system', {})
            if system:
                print(f"\n Sistema:")
                print(f"   RAM média: {system['ram']['avg']:.1f}%")
                print(f"   CPU médio: {system['cpu']['avg']:.1f}%")
        
        print("="*60)
    
    def save_inference_log(self, save_path: str, model_info: Dict = None):
        """Salva log detalhado de inferência"""
        summary = self.get_inference_summary()
        
        if not summary:
            return
        
        # Adicionar informações do modelo se fornecidas
        if model_info:
            summary['model_info'] = model_info
        
        # Adicionar dados detalhados
        detailed_data = {
            'summary': summary,
            'batch_stats': self.batch_stats,
            'raw_times': self.inference_times,
            'batch_sizes': self.batch_sizes,
            'monitoring_config': {
                'log_interval': self.log_interval
            },
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        print(f" Log de inferência salvo: {save_path}")


def create_inference_monitor(log_interval: int = 5) -> InferenceMonitor:
    """
    Cria monitor de inferência
    
    Args:
        log_interval: Intervalo para coleta de dados de hardware
    """
    return InferenceMonitor(log_interval=log_interval)