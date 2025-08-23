from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from buscar_parecidos import buscar_parecidos_manual
from retreinar_com_feedback import retreinar_modelo_ner
import os
import spacy
import logging
import json
import re
from typing import List
from spacy.training import Example
import random
import threading

# --- Configurações Iniciais ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Carregamento de Dados e Modelos ---
try:
    dados = pd.read_csv(
        "materiais.csv",
        sep=";",
        encoding="ISO-8859-1",
        usecols=["CODIGO", "DESCRICAO", "UM", "FAMILIA"],
        on_bad_lines="skip"
    )
    logging.info("Base de dados 'materiais.csv' carregada com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar CSV 'materiais.csv': {e}")
    dados = pd.DataFrame(columns=["CODIGO", "DESCRICAO", "UM", "FAMILIA"])

# --- Caminhos ---
MODEL_PATH = "./treinamento_chat/treinamento_chat_materiais"
FEEDBACK_NER_FILE = "./treinamento_chat/dados_aprendizado.jsonl" 

# Carrega o modelo de PLN na inicialização
nlp = None
try:
    nlp = spacy.load(MODEL_PATH)
    logging.info("Modelo spaCy carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o modelo spaCy: {e}")
    # Trava os endpoints de /chat e /feedback-ner.

# Lock para controlar o acesso ao processo de retreinamento
retraining_lock = threading.Lock()

# --- Classes---
class Material(BaseModel):
    descricao: str
    um: str
    familia: int

class ChatMessage(BaseModel):
    mensagem: str

class EntidadeCorrigida(BaseModel):
    descricao: str
    entidade: str

class FeedbackNER(BaseModel):
    texto_original: str
    entidades_corretas: List[EntidadeCorrigida]

# --- Inicialização da API ---
app = FastAPI(title="API de Materiais", version="1.1")

# --- Endpoint raiz ---
@app.get("/")
def raiz():
    return {"status": "ok", "mensagem": "API de Materiais funcionando!"}

# --- Endpoint para buscar semelhantes ---
@app.post("/buscar")
def buscar(material: Material):
    if dados.empty:
        raise HTTPException(status_code=500, detail="Base de dados não carregada corretamente.")
    try:
        resultados = buscar_parecidos_manual(
            descricao=material.descricao, um=material.um, familia=material.familia,
            dados=dados, top_n=5, score_min=100
        )
        return {"entrada": material.dict(), "resultados": resultados.to_dict(orient="records")}
    except Exception as e:
        logging.error(f"Erro interno no endpoint /buscar: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# --- Endpoint do chat ---
@app.post("/chat")
def chat(chat_message: ChatMessage):
    if nlp is None:
        raise HTTPException(status_code=500, detail="Modelo de linguagem não carregado.")
    
    doc = nlp(chat_message.mensagem)
    entidades_extraidas = {ent.label_: ent.text for ent in doc.ents}
    
    if "DESCRICAO" not in entidades_extraidas:
        return {"status": "erro", "mensagem": "Não consegui identificar a descrição do material na sua mensagem."}
    
    try:
        resultados = buscar_parecidos_manual(
            descricao=entidades_extraidas.get("DESCRICAO"),
            um=entidades_extraidas.get("UM"),
            familia=int(entidades_extraidas.get("FAMILIA")) if entidades_extraidas.get("FAMILIA") else None,
            dados=dados, top_n=5
        )
        return {
            "status": "sucesso", "entrada_chat": chat_message.mensagem,
            "entidades_extraidas": entidades_extraidas, "sugestoes": resultados.to_dict(orient="records")
        }
    except Exception as e:
        logging.error(f"Erro ao processar a requisição do chat: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar sua solicitação.")

# --- Endpoint do feedback ---
@app.post("/feedback-ner")
def salvar_feedback_ner(feedback: FeedbackNER):
    """
    Recebe correções de entidades, formata como dados de treino para o spaCy e
    inicia o processo de re-treinamento.
    """
    texto = feedback.texto_original
    entidades = []
    
    # Encontra os spans (start, end) para cada entidade corrigida
    for ent in feedback.entidades_corretas:
        for match in re.finditer(re.escape(ent.descricao), texto):
            start, end = match.span()
            entidades.append((start, end, ent.entidade))
            break # Pega apenas a primeira ocorrência
    
    if len(entidades) != len(feedback.entidades_corretas):
        raise HTTPException(status_code=400, detail="Não foi possível encontrar todas as entidades no texto original.")

    # Formato de treino do spaCy
    novo_exemplo_treino = (texto, {"entities": entidades})

    # Salva o novo exemplo no arquivo JSONL
    try:
        with open(FEEDBACK_NER_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(novo_exemplo_treino, ensure_ascii=False) + '\n')
        logging.info("Novo exemplo de treino adicionado a partir do feedback.")
    except Exception as e:
        logging.error(f"Erro ao salvar feedback NER: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao salvar o feedback.")
    
    # Inicia o re-treinamento em uma thread separada para não bloquear a resposta da API
    if retraining_lock.acquire(blocking=False): # Tenta adquirir o lock sem bloquear
        try:
            thread = threading.Thread(target=retreinar_modelo_ner)
            thread.start()
            return {"status": "sucesso", "mensagem": "Feedback recebido. Processo de re-treinamento iniciado."}
        finally:
            retraining_lock.release() # Libera o lock após iniciar a thread
    else:
        return {"status": "sucesso", "mensagem": "Feedback recebido. Um processo de re-treinamento já está em andamento."}