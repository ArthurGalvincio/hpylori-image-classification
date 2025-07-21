import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple


class BinaryClassificationMetrics:
    '''Classe para calcular métricas de classificação binária'''
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        '''Calcula todas as métricas'''
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Métricas por classe
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str = None):
        '''Plota matriz de confusão'''
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusão')
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Prevista')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                      save_path: str = None):
        '''Plota curva ROC para classificação binária'''
        if len(self.class_names) != 2:
            print("ROC curve só disponível para classificação binária")
            return
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1])
        auc = roc_auc_score(y_true, y_scores[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        '''Imprime relatório de classificação detalhado'''
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     digits=4)
        print("\nRelatório de Classificação:")
        print("=" * 50)
        print(report)