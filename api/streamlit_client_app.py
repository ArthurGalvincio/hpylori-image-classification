import streamlit as st
import requests
import os
from io import BytesIO
import zipfile
import json
from pathlib import Path

API_BASE_URL = "http://127.0.0.1:8000" 

st.set_page_config(
    page_title="Classificador H.Pylori",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Classificador e Interpretador de H.Pylori")
st.markdown("""
Faça upload de um arquivo ZIP contendo suas imagens (tiles) para classificá-las.
""")

st.warning(f"Certifique-se de que sua API FastAPI está rodando em `{API_BASE_URL}`.")

# --- Opção de Upload de Arquivo ZIP ---
uploaded_zip_file = st.file_uploader(
    "Faça upload de um arquivo ZIP contendo suas imagens (tiles)",
    type=["zip"],
    help="O arquivo ZIP deve conter imagens JPG, JPEG, PNG, etc."
)

if uploaded_zip_file is not None:
    st.subheader("Escolha a Ação:")

    col1, col2 = st.columns(3)

    # --- Opção 1: Visualizar Resultados Direto na API (JSON) ---
    with col1:
        if st.button("📊 Analisar (JSON)"):
            st.info("Enviando ZIP para a API para análise e retorno JSON...")
            
            files = {'zip_file': (uploaded_zip_file.name, uploaded_zip_file.getvalue(), 'application/zip')}
            
            try:
                response = requests.post(f"{API_BASE_URL}/predict_tiles", files=files)
                
                if response.status_code == 200:
                    result_data = response.json()
                    st.success("✅ Análise Concluída!")
                    st.write(f"Status: **{result_data.get('status')}**")
                    st.write(f"Total de Imagens Processadas: **{result_data.get('total_images_processed')}**")
                    
                    if result_data.get('results'):
                        st.subheader("Resultados por Imagem:")
                        st.json(result_data['results'])
                    
                    if result_data.get('errors'):
                        st.error("❌ Erros Encontrados:")
                        st.json(result_data['errors'])
                else:
                    st.error(f"❌ Erro na API: Status {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error("❌ Erro de Conexão: A API FastAPI não está rodando ou o endereço está incorreto.")
            except Exception as e:
                st.exception(f"Ocorreu um erro inesperado: {e}")

    # --- Opção 2: Baixar Resultados como ZIP (JSONs Individuais) ---
    with col2:
        if st.button("📥 Baixar Resultados (ZIP)"):
            st.info("Enviando ZIP para a API para gerar e baixar o ZIP de resultados...")
            
            files = {'zip_file': (uploaded_zip_file.name, uploaded_zip_file.getvalue(), 'application/zip')}
            
            try:
                response = requests.post(f"{API_BASE_URL}/download_json_zip", files=files, stream=True)
                
                if response.status_code == 200:
                    zip_content = BytesIO(response.content)
                    st.success("✅ ZIP de Resultados Gerado!")
                    st.download_button(
                        label="Clique para Baixar o Arquivo ZIP",
                        data=zip_content,
                        file_name="h_pylori_predictions.zip",
                        mime="application/zip"
                    )
                else:
                    st.error(f"❌ Erro na API: Status {response.status_code}")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error("❌ Erro de Conexão: A API FastAPI não está rodando ou o endereço está incorreto.")
            except Exception as e:
                st.exception(f"Ocorreu um erro inesperado: {e}")

