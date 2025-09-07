import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Caminhos dos arquivos
CSV_PATH = "./pesquisa_por_similaridade/materiais.csv"
EMBEDDINGS_PATH = "embeddings.npy"
MODELO_NOME = 'all-MiniLM-L6-v2'

def gerar_e_salvar_embeddings():
    """
    Carrega os materiais do CSV, gera os embeddings das descrições e salva em um arquivo.
    """
    logging.info("Iniciando a geração de embeddings...")
    
    try:
        dados = pd.read_csv(
            CSV_PATH, sep=";", encoding="ISO-8859-1", usecols=["DESCRICAO"]
        )
        logging.info(f"{len(dados)} materiais carregados de '{CSV_PATH}'.")
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo '{CSV_PATH}' não foi encontrado.")
        return

    # Garante que todas as descrições sejam strings
    descricoes = dados['DESCRICAO'].fillna('').astype(str).tolist()
    
    logging.info(f"Carregando o modelo de sentence-transformer: '{MODELO_NOME}'...")
    # O modelo será baixado automaticamente na primeira vez que for usado
    model = SentenceTransformer(MODELO_NOME)
    
    logging.info("Gerando embeddings para as descrições... (Isso pode levar alguns minutos)")
    embeddings = model.encode(descricoes, show_progress_bar=True)
    
    logging.info(f"Embeddings gerados com sucesso. Shape: {embeddings.shape}")
    
    # Salva os embeddings em um arquivo .npy para acesso rápido
    np.save(EMBEDDINGS_PATH, embeddings)
    logging.info(f"Embeddings salvos em '{EMBEDDINGS_PATH}'.")

if __name__ == "__main__":
    gerar_e_salvar_embeddings()