from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from buscar_parecidos import buscar_parecidos_manual
import os
import spacy
import logging

# Configurações de log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializa a API
app = FastAPI(title="API de Materiais", version="1.0")

# Modelo de entrada para busca
class Material(BaseModel):
    descricao: str
    um: str
    familia: int

# Modelo de entrada para feedback
class Feedback(BaseModel):
    entrada: Material
    sugerido: dict   # material retornado pela busca
    correto: bool    # usuário confirma se está certo ou não

# Modelo de entrada para o chat
class ChatMessage(BaseModel):
    mensagem: str

# Carrega a base de dados na inicialização
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

# Caminho para salvar feedbacks
FEEDBACK_FILE = "feedbacks.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["entrada_desc", "entrada_um", "entrada_familia",
                          "sugerido_codigo", "sugerido_desc", "sugerido_um", "sugerido_familia",
                          "correto"]).to_csv(FEEDBACK_FILE, index=False)
    logging.info("Arquivo de feedbacks criado.")

# Carrega o modelo de PLN na inicialização
try:
    nlp = spacy.load("./treinamento_chat/treinamento_chat_materiais")
    logging.info("Modelo spaCy carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o modelo spaCy: {e}")
    nlp = None

# Endpoint raiz
@app.get("/")
def raiz():
    return {"status": "ok", "mensagem": "API de Materiais funcionando!"}

# Endpoint de busca
@app.post("/buscar")
def buscar(material: Material):
    if dados.empty:
        raise HTTPException(status_code=500, detail="Base de dados não carregada corretamente.")

    try:
        resultados = buscar_parecidos_manual(
            descricao=material.descricao,
            um=material.um,
            familia=material.familia,
            dados=dados,
            top_n=5,
            score_min=100
        )

        return {
            "entrada": material.dict(),
            "resultados": resultados.to_dict(orient="records")
        }

    except Exception as e:
        logging.error(f"Erro interno no endpoint /buscar: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# Endpoint para feedback 
@app.post("/feedback")
def salvar_feedback(feedback: Feedback):
    try:
        df_feedback = pd.read_csv(FEEDBACK_FILE)

        novo = {
            "entrada_desc": feedback.entrada.descricao,
            "entrada_um": feedback.entrada.um,
            "entrada_familia": feedback.entrada.familia,
            "sugerido_codigo": feedback.sugerido.get("CODIGO"),
            "sugerido_desc": feedback.sugerido.get("DESCRICAO"),
            "sugerido_um": feedback.sugerido.get("UM"),
            "sugerido_familia": feedback.sugerido.get("FAMILIA"),
            "correto": feedback.correto
        }

        df_feedback = pd.concat([df_feedback, pd.DataFrame([novo])], ignore_index=True)
        df_feedback.to_csv(FEEDBACK_FILE, index=False)
        logging.info("Feedback registrado com sucesso.")

        return {"status": "sucesso", "mensagem": "Feedback registrado com sucesso."}

    except Exception as e:
        logging.error(f"Erro ao salvar feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {str(e)}")

# Endpoint para o chat
@app.post("/chat")
def chat(chat_message: ChatMessage):
    if nlp is None:
        raise HTTPException(status_code=500, detail="Modelo de linguagem não carregado. Verifique os logs de inicialização da API.")

    doc = nlp(chat_message.mensagem)
    entidades_extraidas = {}
    
    for ent in doc.ents:
        entidades_extraidas[ent.label_] = ent.text

    # Verifica se a 'DESCRICAO' foi encontrada. As outras entidades são opcionais.
    if "DESCRICAO" not in entidades_extraidas:
        logging.warning("Não foi possível extrair a descrição. Abortando.")
        return {
            "status": "erro",
            "mensagem": "Não consegui identificar a descrição do material na sua mensagem. A descrição é obrigatória para a busca."
        }
    
    try:
        descricao = entidades_extraidas.get("DESCRICAO")
        # Atribuindo valores padrão caso não sejam encontrados
        um = entidades_extraidas.get("UM")
        familia_str = entidades_extraidas.get("FAMILIA")
        familia = int(familia_str) if familia_str else None

        # Usa a função de busca com as entidades extraídas
        resultados = buscar_parecidos_manual(
            descricao=descricao,
            um=um,
            familia=familia,
            dados=dados,
            top_n=5
        )

        return {
            "status": "sucesso",
            "entrada_chat": chat_message.mensagem,
            "entidades_extraidas": entidades_extraidas,
            "sugestoes": resultados.to_dict(orient="records")
        }

    except ValueError:
        logging.error("Erro de conversão: A família não é um número válido.")
        return {
            "status": "erro",
            "mensagem": "Não consegui identificar a família como um número válido. Verifique se digitou corretamente."
        }
    except Exception as e:
        logging.error(f"Erro ao processar a requisição do chat: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar sua solicitação.")