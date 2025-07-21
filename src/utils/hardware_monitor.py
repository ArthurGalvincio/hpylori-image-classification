import time
import psutil
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class HardwareMonitor:
    """Monitor simples para uso de hardware durante treinamento"""
    
    def __init__(self, log_interval: int = 10):
        """
        Args:
            log_interval: Intervalo em segundos para coleta de dados
        """
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Dados coletados
        self.metrics = {
            'gpu_usage': [],
            'gpu_memory': [],
            'ram_usage': [],
            'cpu_usage': [],
            'timestamps': []
        }
        
        # Controle de tempo
        self.start_time = None
        self.end_time = None
        
        # Inicializar GPU se disponível
        self.gpu_available = False
        self.gpu_count = 0
        self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Inicializa monitoramento de GPU"""
        if NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = True
                print(f" GPU monitoring inicializado: {self.gpu_count} GPU(s) detectada(s)")
            except:
                pass
        
        if not self.gpu_available and GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_count = len(gpus)
                    self.gpu_available = True
                    print(f" GPU monitoring inicializado via GPUtil: {self.gpu_count} GPU(s)")
            except:
                pass
        
        if not self.gpu_available:
            print(" GPU monitoring não disponível (instale: pip install pynvml GPUtil)")
    
    def _get_gpu_stats(self) -> Dict:
        """Coleta estatísticas da GPU"""
        if not self.gpu_available:
            return {'usage': 0, 'memory_used': 0, 'memory_total': 0, 'memory_percent': 0}
        
        try:
            if NVIDIA_AVAILABLE and self.gpu_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
                
                # Uso da GPU
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = utilization.gpu
                
                # Memória
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = memory_info.used / 1024 / 1024
                memory_total_mb = memory_info.total / 1024 / 1024
                memory_percent = (memory_info.used / memory_info.total) * 100
                
                return {
                    'usage': gpu_usage,
                    'memory_used': memory_used_mb,
                    'memory_total': memory_total_mb,
                    'memory_percent': memory_percent
                }
            
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    return {
                        'usage': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                    }
        
        except Exception as e:
            print(f"  Erro ao coletar stats da GPU: {e}")
        
        return {'usage': 0, 'memory_used': 0, 'memory_total': 0, 'memory_percent': 0}
    
    def _get_system_stats(self) -> Dict:
        """Coleta estatísticas do sistema"""
        # RAM
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / 1024 / 1024 / 1024
        ram_total_gb = ram.total / 1024 / 1024 / 1024
        ram_percent = ram.percent
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'ram_used': ram_used_gb,
            'ram_total': ram_total_gb,
            'ram_percent': ram_percent,
            'cpu_percent': cpu_percent
        }
    
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring:
            try:
                # Timestamp
                current_time = time.time()
                
                # Coletar métricas
                gpu_stats = self._get_gpu_stats()
                system_stats = self._get_system_stats()
                
                # Armazenar dados
                self.metrics['timestamps'].append(current_time)
                self.metrics['gpu_usage'].append(gpu_stats['usage'])
                self.metrics['gpu_memory'].append(gpu_stats['memory_percent'])
                self.metrics['ram_usage'].append(system_stats['ram_percent'])
                self.metrics['cpu_usage'].append(system_stats['cpu_percent'])
                
                # Aguardar próximo ciclo
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f" Erro no monitoramento: {e}")
                break
    
    def start_monitoring(self):
        """Inicia o monitoramento"""
        if self.monitoring:
            return
        
        print(" Iniciando monitoramento de hardware...")
        self.start_time = time.time()
        self.monitoring = True
        
        # Resetar métricas
        for key in self.metrics:
            self.metrics[key] = []
        
        # Iniciar thread de monitoramento
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Para o monitoramento"""
        if not self.monitoring:
            return
        
        print(" Parando monitoramento de hardware...")
        self.end_time = time.time()
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
    
    def get_summary(self) -> Dict:
        """Retorna resumo das métricas coletadas"""
        if not self.metrics['timestamps']:
            return {}
        
        # Calcular estatísticas
        def safe_stats(data):
            if not data:
                return {'avg': 0, 'max': 0, 'min': 0}
            return {
                'avg': sum(data) / len(data),
                'max': max(data),
                'min': min(data)
            }
        
        # Tempo total
        total_time = 0
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
        
        summary = {
            'training_time': {
                'total_seconds': total_time,
                'total_minutes': total_time / 60,
                'total_hours': total_time / 3600,
                'formatted': str(timedelta(seconds=int(total_time)))
            },
            'gpu': {
                'available': self.gpu_available,
                'usage': safe_stats(self.metrics['gpu_usage']),
                'memory': safe_stats(self.metrics['gpu_memory'])
            },
            'system': {
                'ram': safe_stats(self.metrics['ram_usage']),
                'cpu': safe_stats(self.metrics['cpu_usage'])
            },
            'data_points': len(self.metrics['timestamps'])
        }
        
        return summary
    
    def print_summary(self):
        """Imprime resumo formatado"""
        summary = self.get_summary()
        
        if not summary:
            print(" Nenhum dado de monitoramento disponível")
            return
        
        print("\n" + "="*60)
        print(" RESUMO DO MONITORAMENTO DE HARDWARE")
        print("="*60)
        
        # Tempo de treinamento
        time_info = summary['training_time']
        print(f"  Tempo Total de Treinamento: {time_info['formatted']}")
        print(f"   ({time_info['total_minutes']:.1f} minutos)")
        
        # GPU
        if summary['gpu']['available']:
            gpu_usage = summary['gpu']['usage']
            gpu_memory = summary['gpu']['memory']
            
            print(f"\n GPU:")
            print(f"   Uso: {gpu_usage['avg']:.1f}% (avg) | {gpu_usage['max']:.1f}% (max)")
            print(f"   VRAM: {gpu_memory['avg']:.1f}% (avg) | {gpu_memory['max']:.1f}% (max)")
        else:
            print(f"\n GPU: Não disponível")
        
        # Sistema
        ram = summary['system']['ram']
        cpu = summary['system']['cpu']
        
        print(f"\n Sistema:")
        print(f"   RAM: {ram['avg']:.1f}% (avg) | {ram['max']:.1f}% (max)")
        print(f"   CPU: {cpu['avg']:.1f}% (avg) | {cpu['max']:.1f}% (max)")
        
        print(f"\n Dados coletados: {summary['data_points']} pontos")
        print("="*60)
    
    def save_detailed_log(self, save_path: str):
        """Salva log detalhado em arquivo JSON"""
        summary = self.get_summary()
        
        if not summary:
            return
        
        # Adicionar dados detalhados
        detailed_data = {
            'summary': summary,
            'raw_metrics': self.metrics,
            'monitoring_config': {
                'log_interval': self.log_interval,
                'gpu_available': self.gpu_available,
                'gpu_count': self.gpu_count
            },
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        print(f" Log detalhado salvo: {save_path}")


def create_simple_monitor(log_interval: int = 30) -> HardwareMonitor:
    """
    Cria um monitor simples de hardware
    
    Args:
        log_interval: Intervalo em segundos para coleta (padrão: 30s)
    
    Returns:
        HardwareMonitor: Instância do monitor
    """
    return HardwareMonitor(log_interval=log_interval)