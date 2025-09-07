import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
import os

# --- Configuração Inicial ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega o modelo e os embeddings uma vez quando o módulo é importado
MODELO_NOME = 'all-MiniLM-L6-v2'
EMBEDDINGS_PATH = "./pesquisa_por_similaridade/embeddings.npy"

try:
    logging.info(f"Carregando modelo '{MODELO_NOME}' para a memória...")
    model = SentenceTransformer(MODELO_NOME)
    logging.info("Modelo carregado.")

    logging.info(f"Carregando embeddings pré-calculados de '{EMBEDDINGS_PATH}'...")
    corpus_embeddings = np.load(EMBEDDINGS_PATH)
    logging.info(f"Embeddings carregados. Shape: {corpus_embeddings.shape}")

except FileNotFoundError:
    logging.error(f"Arquivo de embeddings '{EMBEDDINGS_PATH}' não encontrado.")
    logging.error("Execute o script 'gerar_embeddings.py' primeiro.")
    model = None
    corpus_embeddings = None
except Exception as e:
    logging.error(f"Erro ao carregar modelo ou embeddings: {e}")
    model = None
    corpus_embeddings = None

def buscar_parecidos_semantico(descricao_query: str, um: str, familia: int, dados: pd.DataFrame, top_n=5):
    """
    Busca os materiais mais parecidos usando similaridade semântica (embeddings).
    """
    if model is None or corpus_embeddings is None:
        raise RuntimeError("O modelo de busca semântica não foi carregado corretamente. Verifique os logs.")

    # Gera o embedding para a descrição da busca
    query_embedding = model.encode(descricao_query, convert_to_tensor=True)

    # Calcula a similaridade de cosseno entre a busca e todos os materiais
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_n * 2)
    hits = hits[0]  # A busca é para uma única query

    # Adiciona bônus para UM e Família e recalcula o score
    resultados = []
    indices_vistos = set()

    for hit in hits:
        idx = hit['corpus_id']
        if idx not in indices_vistos:
            
            # Score de similaridade semântica (0 a 1)
            score_semantico = hit['score']
            
            # Pega os dados do material correspondente
            material = dados.iloc[idx]
            
            # Adiciona bônus
            bonus = 0
            if material['FAMILIA'] == familia:
                bonus += 0.2  # Bônus de 20%
            if material['UM'] == um:
                bonus += 0.2  # Bônus de 20%
                
            # Score final, limitado a 1.0
            score_final = min(score_semantico + bonus, 1.0)
            
            resultados.append({
                "CODIGO": material['CODIGO'],
                "DESCRICAO": material['DESCRICAO'],
                "UM": material['UM'],
                "FAMILIA": material['FAMILIA'],
                "SCORE": score_final * 100 # Converte para percentual
            })
            indices_vistos.add(idx)

    # Ordena os resultados pelo score final e retorna o top_n
    df_res = pd.DataFrame(resultados).sort_values(by='SCORE', ascending=False)
    return df_res.head(top_n)