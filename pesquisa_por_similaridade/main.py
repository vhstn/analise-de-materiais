from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
import os
import spacy
import logging
import json
import re
from typing import List
from spacy.training import Example
import random
import threading
from dotenv import load_dotenv
from .buscar_parecidos import buscar_parecidos_semantico as buscar_parecidos
from .retreinar_com_feedback import retreinar_modelo_ner
from .celery_worker import retreinar_modelo_task

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configurações de Segurança ---
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("A variável de ambiente API_KEY não foi definida!")

# Define o esquema de segurança: espera um header chamado "Authorization"
oauth2_scheme = APIKeyHeader(name="Authorization", auto_error=False)

# --- Função de Validação do Token ---
async def validar_token_api(token: str = Security(oauth2_scheme)):
    """
    Valida se o token enviado no header 'Authorization' corresponde ao token.
    O token deve ser enviado no formato 'Bearer <seu_token>'.
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticação não fornecido."
        )
    
    if not token.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Formato do token inválido. Use 'Bearer <token>'."
        )
    
    token_enviado = token.split(" ")[1]

    if token_enviado != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado.",
        )
    return True 

# --- Caminhos ---
MODEL_PATH = "./pesquisa_por_similaridade/treinamento_chat/treinamento_chat_materiais"
FEEDBACK_NER_FILE = "./pesquisa_por_similaridade/treinamento_chat/dados_aprendizado.jsonl" 
CSV_PATH = "./pesquisa_por_similaridade/materiais.csv"

# --- Carregamento de Dados e Modelos ---
try:
    dados = pd.read_csv(
        CSV_PATH,
        sep=";",
        encoding="ISO-8859-1",
        usecols=["CODIGO", "DESCRICAO", "UM", "FAMILIA"],
        on_bad_lines="skip"
    )
    logging.info("Base de dados 'materiais.csv' carregada com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar CSV 'materiais.csv': {e}")
    dados = pd.DataFrame(columns=["CODIGO", "DESCRICAO", "UM", "FAMILIA"])

# --- Gerenciador de Modelo ---
class ModelManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.nlp = None
        self.load_model()

    def load_model(self):
        try:
            self.nlp = spacy.load(self.model_path)
            logging.info("Modelo spaCy carregado/recarregado com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao carregar o modelo spaCy: {e}")
            self.nlp = None
    
    def get_model(self):
        return self.nlp

model_manager = ModelManager(MODEL_PATH)

# Carrega o modelo de PLN na inicialização
nlp = None
try:
    nlp = spacy.load(MODEL_PATH)
    logging.info("Modelo spaCy carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o modelo spaCy: {e}")

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
app = FastAPI(title="API de Materiais", version="1.3-secure")

# --- Endpoint raiz ---
@app.get("/")
def raiz():
    return {"status": "ok", "mensagem": "API de Materiais funcionando!"}

# --- Endpoint para buscar semelhantes ---
@app.post("/buscar", dependencies=[Depends(validar_token_api)])
def buscar(material: Material):
    if dados.empty:
        raise HTTPException(status_code=500, detail="Base de dados não carregada corretamente.")
    try:
        resultados = buscar_parecidos(
            descricao_query=material.descricao, um=material.um, familia=material.familia,
            dados=dados, top_n=5
        )
        return {"entrada": material.dict(), "resultados": resultados.to_dict(orient="records")}
    except Exception as e:
        logging.error(f"Erro interno no endpoint /buscar: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# --- Endpoint do chat ---
@app.post("/chat", dependencies=[Depends(validar_token_api)])
def chat(chat_message: ChatMessage):
    if nlp is None:
        raise HTTPException(status_code=500, detail="Modelo de linguagem não carregado.")
    
    doc = nlp(chat_message.mensagem)
    entidades_extraidas = {ent.label_: ent.text for ent in doc.ents}
    
    if "DESCRICAO" not in entidades_extraidas:
        return {"status": "erro", "mensagem": "Não consegui identificar a descrição do material na sua mensagem."}
    
    try:
        familia_str = entidades_extraidas.get("FAMILIA")
        familia = int(familia_str) if familia_str else None
        
        resultados = buscar_parecidos(
            descricao_query=entidades_extraidas.get("DESCRICAO"),
            um=entidades_extraidas.get("UM", ""), # Garante um valor padrão
            familia=familia,
            dados=dados, 
            top_n=5
        )
        return {
            "status": "sucesso", "entrada_chat": chat_message.mensagem,
            "entidades_extraidas": entidades_extraidas, "sugestoes": resultados.to_dict(orient="records")
        }
    except Exception as e:
        logging.error(f"Erro ao processar a requisição do chat: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar sua solicitação.")

# --- Endpoint do feedback ---
@app.post("/feedback-ner", dependencies=[Depends(validar_token_api)])
def salvar_feedback_ner(feedback: FeedbackNER):
    """
    Recebe correções de entidades, formata como dados de treino para o spaCy e
    inicia o processo de re-treinamento.
    """
    texto = feedback.texto_original
    entidades = []
    
    # Encontra os spans (start, end) para cada entidade corrigida
    for ent in feedback.entidades_corretas:
        for match in re.finditer(re.escape(ent.descricao), texto, flags=re.IGNORECASE):
            start, end = match.span()
            entidades.append((start, end, ent.entidade))
            break
    
    if len(entidades) != len(feedback.entidades_corretas):
        raise HTTPException(status_code=400, detail="Não foi possível encontrar todas as entidades no texto original.")

    # Formato de treino do spaCy
    novo_exemplo_treino = (texto, {"entities": entidades})

    try:
        with open(FEEDBACK_NER_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(novo_exemplo_treino, ensure_ascii=False) + '\n')
        logging.info("Novo exemplo de treino adicionado. Enviando tarefa para a fila.")

        # Envia a tarefa para o Celery em vez de usar uma thread
        retreinar_modelo_task.delay()

        return {"status": "sucesso", "mensagem": "Feedback recebido. O retreinamento foi agendado e ocorrerá em segundo plano."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar ou agendar feedback: {e}")
