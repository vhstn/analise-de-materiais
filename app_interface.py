import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json

# Carrega as variáveis de ambiente (para pegar a chave da API)
load_dotenv()

# --- Configurações da Aplicação ---
API_URL = "http://api:8000"
API_KEY = os.getenv("API_KEY")

st.set_page_config(page_title="Análise de Materiais", layout="centered")

# --- Funções de Comunicação com a API ---

def buscar_materiais_chat(mensagem_chat: str):
    """Envia uma mensagem para o endpoint /chat da API."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"mensagem": mensagem_chat}
    try:
        response = requests.post(f"{API_URL}/chat", headers=headers, json=payload)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Erro ao conectar com a API: {e}"

def buscar_materiais_direto(descricao: str, um: str, familia: int):
    """Envia uma busca direta para o endpoint /buscar da API."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"descricao": descricao, "um": um, "familia": familia}
    try:
        response = requests.post(f"{API_URL}/buscar", headers=headers, json=payload)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Erro ao conectar com a API: {e}"


def enviar_feedback(texto_original: str, entidades: list):
    """Envia feedback para o endpoint /feedback-ner da API."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"texto_original": texto_original, "entidades_corretas": entidades}
    try:
        response = requests.post(f"{API_URL}/feedback-ner", headers=headers, json=payload)
        response.raise_for_status()
        return True, response.json().get("mensagem", "Feedback enviado com sucesso.")
    except requests.exceptions.HTTPError as e:
        try:
            return False, e.response.json().get("detail", str(e))
        except json.JSONDecodeError:
            return False, str(e)
    except requests.exceptions.RequestException as e:
        return False, f"Erro de conexão: {e}"

# --- Interface Gráfica (UI) ---

st.title("🤖 Assistente de Análise de Materiais")
st.markdown("Digite o que você precisa e o assistente buscará os materiais mais parecidos.")

# Inicializa o estado da sessão
if 'busca_realizada' not in st.session_state:
    st.session_state.busca_realizada = False
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
    st.session_state.busca_realizada = True
    if st.session_state.chat_mensagem:
        with st.spinner("Buscando na base de materiais..."):
            resposta, erro = buscar_materiais_chat(st.session_state.chat_mensagem)
            
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
        st.session_state.busca_realizada = False


# --- Seção de Resultados e Feedback ---

if st.session_state.busca_realizada:
    # Exibe erro, se houver
    if st.session_state.erro:
        st.error(f"Erro na busca: {st.session_state.erro}")

    # Exibe entidades extraídas
    if st.session_state.entidades:
        st.markdown("---")
        st.subheader("🔍 Entidades Extraídas:")
        cols = st.columns(len(st.session_state.entidades))
        for i, (label, texto) in enumerate(st.session_state.entidades.items()):
            with cols[i]:
                st.metric(label=label, value=str(texto)) # Garante que o valor seja string

    # Exibe sugestões
    if st.session_state.resultados:
        st.markdown("---")
        st.subheader("✅ Sugestões Encontradas:")
        for item in st.session_state.resultados:
            with st.expander(f"**{item['DESCRICAO']}** (Score: {item['SCORE']:.2f})"):
                st.markdown(f"**Código:** `{item['CODIGO']}`")
                st.markdown(f"**Família:** `{item['FAMILIA']}`")
                st.markdown(f"**UM:** `{item['UM']}`")

    # --- Formulário de Feedback em um Expander ---
    st.markdown("---")
    with st.expander("A extração está incorreta? Clique aqui para corrigir."):
        with st.form("feedback_form"):
            st.info("Preencha os campos com os valores corretos. Apenas a descrição é obrigatória.")
            
            # Preenche o formulário com as entidades que o modelo encontrou
            entidades_atuais = st.session_state.entidades or {}
            
            descricao_correta = st.text_input("Descrição Correta", value=entidades_atuais.get("DESCRICAO", ""))
            familia_correta = st.text_input("Família Correta", value=entidades_atuais.get("FAMILIA", ""))
            um_correta = st.text_input("UM Correta", value=entidades_atuais.get("UM", ""))
            
            submit_feedback = st.form_submit_button("Enviar Feedback e Refazer Busca")

        if submit_feedback:
            entidades_feedback = []
            if descricao_correta:
                entidades_feedback.append({"descricao": descricao_correta, "entidade": "DESCRICAO"})
            if familia_correta:
                entidades_feedback.append({"descricao": familia_correta, "entidade": "FAMILIA"})
            if um_correta:
                entidades_feedback.append({"descricao": um_correta, "entidade": "UM"})

            if not descricao_correta:
                st.warning("A descrição é obrigatória para o feedback.")
            else:
                with st.spinner("Enviando feedback e retreinando o modelo..."):
                    sucesso, mensagem = enviar_feedback(st.session_state.chat_mensagem, entidades_feedback)
                    if sucesso:
                        st.success(f"Feedback enviado! Mensagem da API: '{mensagem}'")
                    else:
                        st.error(f"Erro ao enviar feedback: {mensagem}")

                # --- Refaz a busca com os dados corrigidos ---
                if sucesso:
                    with st.spinner("Refazendo a busca com os dados corrigidos..."):
                        try:
                            # Valida se a família é um inteiro antes de buscar
                            familia_int = int(familia_correta) if familia_correta.isdigit() else 0
                            
                            resposta_direta, erro_direto = buscar_materiais_direto(descricao_correta, um_correta, familia_int)
                            if erro_direto:
                                st.error(erro_direto)
                            else:
                                st.info("Resultados atualizados com base na sua correção:")
                                # Atualiza o estado da sessão para forçar o rerender da tela com os novos resultados
                                st.session_state.resultados = resposta_direta.get("resultados", [])
                                st.session_state.entidades = {"DESCRICAO": descricao_correta, "UM": um_correta, "FAMILIA": familia_correta}
                                st.rerun() # Força a re-renderização da página
                        except Exception as e:
                            st.error(f"Ocorreu um erro ao refazer a busca: {e}")