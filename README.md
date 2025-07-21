# 🔬 Sistema de Classificação de Imagens Médicas

Este projeto é um modelo inicial de framework para treinar, avaliar e implementar modelos de Deep Learning em tarefas de classificação binária de imagens médicas. Ideal para pesquisadores e desenvolvedores que buscam uma solução robusta e customizável.

**Nota:** Um relatório detalhado referente a este projeto, desenvolvido para a disciplina de **Introdução à Inteligência Artificial**, está disponível no repositório.

## 🚀 Instalação e Setup

1.  **Clonar o Repositório:**
    ```bash
    git clone <https://github.com/ArthurGalvincio/hpylori-image-classification>
    cd classifier
    ```

2.  **Criar e Ativar Ambiente Virtual:**
    ```bash
    python -m venv my-env
    # No Windows PowerShell:
    .\my-env\Scripts\activate
    # No Linux/macOS:
    source my-env/bin/activate
    ```

3.  **Instalar Dependências:**
    * Crie um arquivo `requirements.txt` na raiz do projeto (`classifier/`) com as seguintes dependências:
        ```
        torch
        torchvision
        pyyaml
        tqdm
        scikit-learn
        matplotlib
        seaborn
        psutil
        pynvml
        GPUtil
        fastapi
        uvicorn
        python-multipart
        requests
        streamlit
        pydantic 
        Pillow-SIMD 
        ```
    * Instale as dependências:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Criar Estrutura de Diretórios:**
    * **Crie os diretórios necessários:**
    ```bash
    mkdir -p data/processed data/raw
    mkdir -p models/saved
    mkdir -p src/indices
    mkdir -p logs
    mkdir -p results
    mkdir -p api
    ```

5.  **Configurar Seu Dataset:**
    * Este sistema espera um dataset de imagens histopatológicas da mucosa gástrica, onde as imagens são organizadas em subpastas por classe (ex: `positive/` e `negative/`).
    * **Exemplo de Estrutura:**
        ```
        seu_dataset_personalizado/
        ├── negative/
        │   ├── img_neg_001.jpg
        │   └── ...
        └── positive/
            ├── img_pos_001.jpg
            └── ...
        ```
    * **Configuração:** Atualize o `data.root_path` em seus arquivos `configs/*.yaml` (ex: `configs/inception_v3_config.yaml`) para apontar para o caminho do seu dataset local. Ex: `root_path: "data/processed/dataset".

## 👨‍💻 Como Usar o Sistema

### **Treinar Modelos**

* **Ajuste os arquivos de configuração (`configs/*.yaml`):**
    * Defina `data.root_path` para o seu dataset.
    * Ajuste `training.epochs`, `training.learning_rate`, `dataloader.batch_size` e outras configurações conforme sua necessidade.
* **Execute o treinamento:**
    ```bash
    python -m src.main.train --config configs/seu_modelo_config.yaml
    ```
* **Retomar Treino de Checkpoint (em caso de interrupção):**
    ```bash
    python -m src.main.train --config configs/seu_modelo_config.yaml --resume_from models/saved/seu_modelo/best_model.pth
    ```

### **Avaliar e Predizer Modelos**

* **Avaliar Modelo Treinado (com métricas detalhadas):**
    * Os resultados (métricas e performance de inferência) serão salvos no arquivo JSON especificado por `--output`.
    ```bash
    python -m src.main.evaluate \
        --checkpoint models/saved/seu_modelo/best_model.pth \
        --data data/processed/seu_dataset_de_teste \
        --device cuda \
        --output results/seu_modelo/avaliacao_teste_gpu.json
    ```
* **Fazer Predições em Imagens/Diretórios:**
    ```bash
    python -m src.main.predict --checkpoint models/saved/seu_modelo/best_model.pth --image path/to/image.jpg
    ```

### **Usar API e Interface Gráfica (Streamlit)**

1.  **Iniciar a API FastAPI (Backend - em um terminal SEPARADO):**
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    * Deixe este terminal aberto e a API rodando. Ela estará acessível em `http://127.0.0.1:8000` (local) e em `http://[SEU_IP_NA_REDE]:8000` (para outros PCs na mesma rede).

2.  **Iniciar o Aplicativo Streamlit (Frontend - em outro terminal SEPARADO):**
    ```bash
    streamlit run streamlit_client_app.py
    ```
    * Isso abrirá uma aba no seu navegador (`http://localhost:8501`). A interface Streamlit atuará como cliente da sua API FastAPI.
    * Após o deploy, o aplicativo Streamlit estará acessível em: [URL_DO_SEU_STREAMLIT_AQUI]

3.  **Uso:**
    * Para testar a funcionalidade de upload e visualização/download de predições, um arquivo `tiles_teste.zip` (contendo 5 imagens `positive` e 5 `negative` de exemplo) está incluído na raiz do repositório.
    * Acesse o Streamlit em seu navegador (`http://localhost:8501` ou a URL de deploy disponível na coluna da direita do repositório).
    * Faça upload do arquivo `tiles_teste.zip` (ou outro arquivo ZIP de imagens).
    * Escolha entre visualizar os resultados JSON ou baixar o ZIP com JSONs individuais.
