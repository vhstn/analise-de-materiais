from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from buscar_parecidos import buscar_parecidos_manual
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
    # A API pode continuar funcionando para outras rotas, mas /chat e /feedback-ner falharão.

# Lock para controlar o acesso ao processo de retreinamento
retraining_lock = threading.Lock()

# --- Funções Auxiliares e de Treinamento ---

def retreinar_modelo_ner():
    """
    Carrega o modelo spaCy, lê os novos dados de feedback e atualiza o componente NER.
    """
    global nlp
    if not os.path.exists(FEEDBACK_NER_FILE):
        logging.info("Nenhum novo dado de feedback para treinar.")
        return

    try:
        logging.info("Iniciando o processo de re-treinamento do modelo NER.")
        
        # Carrega o modelo do disco para garantir que estamos treinando a versão mais recente
        nlp_to_train = spacy.load(MODEL_PATH)
        ner = nlp_to_train.get_pipe("ner")

        # Carrega os novos dados de treino do arquivo de feedback
        train_examples = []
        with open(FEEDBACK_NER_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data[0]
                annotations = data[1]
                train_examples.append(Example.from_dict(nlp_to_train.make_doc(text), annotations))

        if not train_examples:
            logging.info("Arquivo de feedback encontrado, mas está vazio. Nenhum treinamento necessário.")
            return

        logging.info(f"Re-treinando com {len(train_examples)} novos exemplos.")

        # Desativa outros pipes para focar no NER
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp_to_train.pipe_names if pipe not in pipe_exceptions]
        
        with nlp_to_train.disable_pipes(*other_pipes):
            optimizer = nlp_to_train.resume_training()
            for i in range(10):  # Treinamos por um número menor de iterações para atualizações
                random.shuffle(train_examples)
                losses = {}
                nlp_to_train.update(train_examples, drop=0.35, losses=losses, sgd=optimizer)
                logging.info(f"Iteração de re-treino {i+1}/10 - Perda: {losses.get('ner', 0.0):.4f}")

        # Salva o modelo atualizado
        nlp_to_train.to_disk(MODEL_PATH)
        logging.info(f"Modelo re-treinado e salvo com sucesso em: {MODEL_PATH}")

        # Recarrega o modelo global na memória para que a API use a versão atualizada
        nlp = spacy.load(MODEL_PATH)
        logging.info("Modelo global em memória atualizado para a nova versão.")
        
        # Opcional: Arquivar ou limpar o arquivo de feedback após o treino
        os.rename(FEEDBACK_NER_FILE, f"{FEEDBACK_NER_FILE}.processed_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}")
    except Exception as e:
        logging.error(f"Erro durante o re-treinamento: {e}")

# --- Classes---
class Material(BaseModel):
    descricao: str
    um: str
    familia: int

class ChatMessage(BaseModel):
    mensagem: str

class EntidadeCorrigida(BaseModel):
    texto: str
    label: str

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
        for match in re.finditer(re.escape(ent.texto), texto):
            start, end = match.span()
            entidades.append((start, end, ent.label))
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