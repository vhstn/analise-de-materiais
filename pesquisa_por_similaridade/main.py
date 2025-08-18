from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from buscar_parecidos import buscar_parecidos_manual
import os

# Inicializa a API
app = FastAPI(title="API de Materiais", version="1.0")

# Modelo de entrada
class Material(BaseModel):
    descricao: str
    um: str
    familia: int

# Modelo de entrada para feedback
class Feedback(BaseModel):
    entrada: Material
    sugerido: dict   # material retornado pela busca
    correto: bool    # usuário confirma se está certo ou não

#  Carrega a base
try:
    dados = pd.read_csv(
        "materiais.csv",
        sep=";",
        encoding="ISO-8859-1",
        usecols=["CODIGO", "DESCRICAO", "UM", "FAMILIA"],
        on_bad_lines="skip"
    )
except Exception as e:
    print(f" Erro ao carregar CSV: {e}")
    dados = pd.DataFrame(columns=["CODIGO", "DESCRICAO", "UM", "FAMILIA"])

# Caminho para salvar feedbacks
FEEDBACK_FILE = "feedbacks.csv"
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["entrada_desc", "entrada_um", "entrada_familia",
                          "sugerido_codigo", "sugerido_desc", "sugerido_um", "sugerido_familia",
                          "correto"]).to_csv(FEEDBACK_FILE, index=False)

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
            top_n=5
        )

        return {
            "entrada": material.dict(),
            "resultados": resultados.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# Endpoint para feedback
@app.post("/feedback")
def salvar_feedback(feedback: Feedback):
    try:
        # Carrega feedback existente
        df_feedback = pd.read_csv(FEEDBACK_FILE)

        # Adiciona novo registro
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

        return {"status": "sucesso", "mensagem": "Feedback registrado com sucesso."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {str(e)}")