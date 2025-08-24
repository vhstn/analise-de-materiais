import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (para pegar a chave da API)
load_dotenv()

# --- Configurações da Aplicação ---
API_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("API_KEY")

st.set_page_config(page_title="Análise de Materiais", layout="centered")

# --- Funções de Comunicação com a API ---

def buscar_materiais(mensagem_chat: str):
    """Envia uma mensagem para o endpoint /chat da API e retorna os resultados."""
    if not API_KEY:
        st.error("Chave da API não encontrada! Verifique seu arquivo .env.")
        return None, None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "mensagem": mensagem_chat
    }

    try:
        response = requests.post(f"{API_URL}/chat", headers=headers, json=payload)
        response.raise_for_status()  # Lança um erro para respostas com status 4xx ou 5xx
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Erro ao conectar com a API: {e}"

def enviar_feedback(texto_original: str, entidades: list):
    """Envia feedback para o endpoint /feedback-ner da API."""
    if not API_KEY:
        st.error("Chave da API não encontrada! Verifique seu arquivo .env.")
        return False, "Chave da API não encontrada."

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "texto_original": texto_original,
        "entidades_corretas": entidades
    }

    try:
        response = requests.post(f"{API_URL}/feedback-ner", headers=headers, json=payload)
        response.raise_for_status()
        return True, response.json().get("mensagem", "Feedback enviado com sucesso.")
    except requests.exceptions.RequestException as e:
        return False, f"Erro ao enviar feedback para a API: {e}"

# --- Interface Gráfica (UI) ---

st.title("🤖 Assistente de Análise de Materiais")
st.markdown("Digite o que você precisa e o assistente buscará os materiais mais parecidos.")

# Inicializa o estado da sessão
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'entidades' not in st.session_state:
    st.session_state.entidades = None
if 'erro' not in st.session_state:
    st.session_state.erro = None
if 'chat_mensagem' not in st.session_state:
    st.session_state.chat_mensagem = ""

# Campo de entrada de texto
st.session_state.chat_mensagem = st.text_input(
    "O que você está buscando?", 
    placeholder="Ex: Preciso de um parafuso sextavado M8",
    value=st.session_state.chat_mensagem
)

if st.button("Buscar"):
    if st.session_state.chat_mensagem:
        with st.spinner("Buscando na base de materiais..."):
            resposta, erro = buscar_materiais(st.session_state.chat_mensagem)
            
            if erro:
                st.session_state.erro = erro
                st.session_state.resultados = None
                st.session_state.entidades = None
            elif resposta and resposta.get("status") == "sucesso":
                st.session_state.resultados = resposta.get("sugestoes", [])
                st.session_state.entidades = resposta.get("entidades_extraidas", {})
                st.session_state.erro = None
            else:
                st.session_state.erro = resposta.get("mensagem", "Ocorreu um erro desconhecido.")
                st.session_state.resultados = None
                st.session_state.entidades = None
    else:
        st.warning("Por favor, digite algo para buscar.")

# Exibição dos resultados e do formulário de feedback
if st.session_state.erro:
    st.error(st.session_state.erro)
    
    # Se o erro for sobre a falta de descrição, exibe o formulário de feedback
    if "Não consegui identificar a descrição" in st.session_state.erro:
        st.markdown("---")
        st.subheader("💡 Ajude o assistente a aprender!")
        st.info("Não consegui entender a descrição do material. Por favor, corrija a informação abaixo:")

        with st.form("feedback_form"):
            texto_original = st.text_input("Frase Original", value=st.session_state.chat_mensagem, disabled=True)
            
            descricao = st.text_input("Qual é a **descrição** correta?")
            familia = st.text_input("Qual a **família** (opcional)?")
            um = st.text_input("Qual a **unidade de medida (UM)** (opcional)?")
            
            submit_feedback = st.form_submit_button("Enviar Feedback")

        if submit_feedback:
            entidades_corretas = []
            if descricao:
                entidades_corretas.append({"descricao": descricao, "entidade": "DESCRICAO"})
            if familia:
                entidades_corretas.append({"descricao": familia, "entidade": "FAMILIA"})
            if um:
                entidades_corretas.append({"descricao": um, "entidade": "UM"})

            if entidades_corretas:
                sucesso, mensagem = enviar_feedback(texto_original, entidades_corretas)
                if sucesso:
                    st.success(f"Feedback enviado com sucesso! O modelo irá aprender com sua correção. Mensagem da API: '{mensagem}'")
                else:
                    st.error(f"Erro ao enviar feedback: {mensagem}")
            else:
                st.warning("Preencha ao menos a descrição para enviar o feedback.")


if st.session_state.entidades:
    st.markdown("---")
    st.subheader("🔍 Entidades Extraídas da sua Mensagem:")
    
    cols = st.columns(len(st.session_state.entidades))
    for i, (label, texto) in enumerate(st.session_state.entidades.items()):
        with cols[i]:
            st.metric(label=label, value=texto)

if st.session_state.resultados:
    st.markdown("---")
    st.subheader("✅ Sugestões Encontradas:")
    
    for item in st.session_state.resultados:
        with st.expander(f"**{item['DESCRICAO']}** (Score: {item['SCORE']:.2f})"):
            st.markdown(f"**Código:** `{item['CODIGO']}`")
            st.markdown(f"**Família:** `{item['FAMILIA']}`")
            st.markdown(f"**UM:** `{item['UM']}`")