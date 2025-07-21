import json
import os
import shutil
import sys
import zipfile
import tempfile
import io
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models import create_model
from src.utils import get_val_transforms
from src.main.shap_utils import generate_shap_explanation


app = FastAPI(
    title="H.Pylori Classification API",
    description="API para classificar imagens (tiles) de H.Pylori como positive ou negative, com opções de retorno JSON visual ou ZIP de JSONs.",
    version="1.0.0"
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Caminho para o seu melhor modelo (ajuste conforme a sua estrutura real)
MODEL_PATH = Path(__file__).parent.parent / "models" / "saved" / "inception_v3" / "best_model.pth"

model: Optional[nn.Module] = None
model_config: Optional[Dict] = None
model_transform = None
class_names: Optional[List[str]] = None
device: torch.device = torch.device("cuda") 
'''
Device: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device
'''

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Carrega modelo do checkpoint."""
    print(f"Carregando modelo do checkpoint: {checkpoint_path} no dispositivo {device}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        loaded_model = create_model(config)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.to(device)
        loaded_model.eval()
        
        print(f"Modelo '{config['model']['name']}' carregado com sucesso!")
        return loaded_model, config
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao carregar o modelo: {e}")

@app.on_event("startup")
async def startup_event():
    global model, model_config, model_transform, class_names, device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU disponível: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("GPU não disponível, usando CPU.")
    try:
        model, model_config = load_model_from_checkpoint(str(MODEL_PATH), device)

        img_size = model_config.get('model', {}).get('img_size', 224)
        if 'inception' in model_config.get('model', {}).get('name', '').lower():
            print("Modelo Inception detectado. Forçando img_size para 299.")
            img_size = 299

        model_transform = get_val_transforms(img_size=img_size)
        class_names = model_config['data']['class_names']
        print("API pronta para receber requisições.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha crítica na inicialização da API: {e}")

# --- Definição do Formato de Resposta ---
class PredictionResult(BaseModel):
    filename: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str

# ---Modelo de resposta que inclui a URL da imagem SHAP ---
class ShapPredictionResult(PredictionResult):
    shap_image_url: Optional[str] = None
    shap_error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    status: str
    total_images_processed: int
    results: List[PredictionResult]
    errors: List[Dict[str, str]] = []

class ShapBatchResponse(BaseModel):
    status: str
    total_images_processed: int
    results: List[ShapPredictionResult]
    errors: List[Dict[str, str]] = []

# --- Lógica de Predição Compartilhada ---
async def process_images_for_prediction(image_paths: List[Path]) -> Tuple[List[PredictionResult], List[Dict]]:
    """
    Função auxiliar para processar uma lista de caminhos de imagem e obter predições.
    Retorna uma lista de PredictionResult e uma lista de erros.
    """
    results: List[PredictionResult] = []
    errors: List[Dict[str, str]] = []

    if model is None or model_transform is None or class_names is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado. A API não foi inicializada corretamente.")

    model.eval() 
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # Filtrar arquivos não imagem ou vazios
                if not img_path.is_file() or img_path.stat().st_size == 0:
                    errors.append({"filename": img_path.name, "error": "Arquivo não é uma imagem válida ou está vazio."})
                    continue

                image = Image.open(img_path).convert("RGB")
                image_tensor = model_transform(image).unsqueeze(0).to(device)

                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)

                predicted_class_name = class_names[predicted_idx.item()]

                results.append(PredictionResult(
                    filename=img_path.name,
                    prediction=f"H.Pylori: {predicted_class_name}",
                    confidence=confidence.item(),
                    probabilities={name: prob.item() for name, prob in zip(class_names, probs[0])},
                    timestamp=datetime.now().isoformat() 
                ))
            except Exception as e:
                errors.append({"filename": img_path.name, "error": str(e)})
                print(f"Erro ao processar imagem {img_path.name}: {e}")
    return results, errors

# --- Endpoint 1: Retorno JSON Visual Direto ---
@app.post("/predict_tiles", response_model=BatchPredictionResponse)
async def predict_tiles_json_response(zip_file: UploadFile = File(...)):
    """
    Recebe um arquivo ZIP com imagens (tiles) e retorna uma resposta JSON estruturada
    para visualização direta no navegador.
    """
    if not zip_file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Formato de arquivo inválido. Por favor, envie um arquivo ZIP.")

    temp_input_dir = None
    processed_count = 0
    
    try:
        with tempfile.TemporaryDirectory() as tmp_base_dir:
            temp_zip_path = Path(tmp_base_dir) / zip_file.filename
            with open(temp_zip_path, "wb") as buffer:
                shutil.copyfileobj(zip_file.file, buffer)

            extracted_images_dir = Path(tmp_base_dir) / "extracted_images"
            extracted_images_dir.mkdir()
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_images_dir)
            
            print(f"Arquivo ZIP '{zip_file.filename}' descompactado para {extracted_images_dir}")

            image_paths = []
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
            for root, _, files in os.walk(extracted_images_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_paths.append(Path(root) / file)
            
            if not image_paths:
                raise HTTPException(status_code=400, detail="Nenhuma imagem válida encontrada no arquivo ZIP de entrada.")

            results, errors = await process_images_for_prediction(image_paths) 
            processed_count = len(results) + len([e for e in errors if "filename" in e]) 

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Arquivo ZIP corrompido ou inválido.")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Erro inesperado no servidor: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")
    finally:
            print(f"Processamento concluído para {zip_file.filename}. Imagens processadas: {processed_count}")

    return BatchPredictionResponse(
        status="success" if not errors else "partial success",
        total_images_processed=processed_count,
        results=results,
        errors=errors
    )

# --- Endpoint 2: Retorno ZIP com JSONs Individuais ---
@app.post("/download_json_zip", 
          response_description="ZIP file containing JSON results for each image")
async def download_json_zip_response(zip_file: UploadFile = File(...)):
    """
    Recebe um arquivo ZIP contendo múltiplas imagens (tiles) e retorna um arquivo ZIP
    contendo um JSON para cada tile com seu resultado de classificação.
    """
    if not zip_file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Formato de arquivo inválido. Por favor, envie um arquivo ZIP.")

    temp_input_dir = None
    temp_output_json_dir = None
    output_zip_path = None

    try:
        # 1. Salvar o arquivo ZIP de entrada e descompactar
        temp_input_dir = Path(tempfile.mkdtemp())
        temp_zip_path = temp_input_dir / zip_file.filename
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(zip_file.file, buffer)

        extracted_images_dir = temp_input_dir / "extracted_images"
        extracted_images_dir.mkdir()
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_images_dir)
        
        image_paths = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
        for root, _, files in os.walk(extracted_images_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(Path(root) / file)
        
        if not image_paths:
            raise HTTPException(status_code=400, detail="Nenhuma imagem válida encontrada no arquivo ZIP de entrada.")

        # 2. Processar imagens e salvar resultados como JSONs individuais
        temp_output_json_dir = Path(tempfile.mkdtemp()) # Novo diretório para os JSONs de saída
        results_list, errors_list = await process_images_for_prediction(image_paths) # Usa a função compartilhada

        for result in results_list:
            output_json_filename = Path(result.filename).stem + ".json"
            with open(temp_output_json_dir / output_json_filename, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(mode='json'), f, indent=2) # Converte Pydantic model para dict/JSON

        # 3. Criar o arquivo ZIP de saída com todos os JSONs
        output_zip_filename = f"predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"

        output_zip_base_path = Path(tempfile.gettempdir()) / output_zip_filename
        
        with zipfile.ZipFile(output_zip_base_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_output_json_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_output_json_dir))
        
        # 4. Preparar o StreamingResponse
        response_stream = io.BytesIO()
        with open(output_zip_base_path, "rb") as f:
            response_stream.write(f.read())
        response_stream.seek(0)

        headers = {
            'Content-Disposition': f'attachment; filename="{output_zip_filename}"',
            'Content-Type': 'application/zip'
        }
        return StreamingResponse(response_stream, headers=headers, media_type="application/zip")

    except HTTPException as e:
        raise e
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Arquivo ZIP corrompido ou inválido.")
    except Exception as e:
        print(f"Erro inesperado no servidor: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")
    finally:
        # Limpar diretórios temporários
        if temp_input_dir and temp_input_dir.exists(): shutil.rmtree(temp_input_dir)
        if temp_output_json_dir and temp_output_json_dir.exists(): shutil.rmtree(temp_output_json_dir)
        if output_zip_path and output_zip_path.exists(): os.remove(output_zip_path) # Remover o arquivo ZIP temporário