import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List
import seaborn as sns


def visualizar_amostras(dataset, num_samples=8, figsize=(12, 6)):
    '''Visualiza amostras aleatórias do dataset'''
    
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.ravel()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose(1, 2, 0)
            # Desnormalizar se necessário
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        # Tratar label que pode ser int ou dict
        if isinstance(label, dict):
            title = f"Classe: {label.get('label', 'Unknown')}"
        else:
            if hasattr(dataset, 'classes'):
                title = f"Classe: {dataset.classes[label]}"
            else:
                title = f"Classe: {label}"
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_class_distribution(datasets_dict: dict, figsize=(10, 6)):
    '''Plota distribuição de classes para múltiplos datasets'''
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data_for_plot = []
    
    for name, dataset in datasets_dict.items():
        class_counts = {}
        
        for _, label in dataset:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        for class_idx, count in class_counts.items():
            data_for_plot.append({
                'Dataset': name,
                'Classe': class_idx,
                'Contagem': count
            })
    
    # Converter para formato adequado para plot
    import pandas as pd
    df = pd.DataFrame(data_for_plot)
    
    # Criar gráfico de barras agrupadas
    df_pivot = df.pivot(index='Classe', columns='Dataset', values='Contagem')
    df_pivot.plot(kind='bar', ax=ax)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Número de Imagens')
    ax.set_title('Distribuição de Classes por Dataset')
    ax.legend(title='Dataset')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def calcular_estatisticas_dataset(dataloader, num_batches=None):
    '''Calcula média e desvio padrão do dataset'''
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for i, (data, _) in enumerate(dataloader):
        if num_batches and i >= num_batches:
            break
            
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean, std