# Análise de materiais

Este projeto contém duas funções para análise de materiais:

1. Sugestão de materiais mais parecidos — Dada uma descrição, unidade de medida e família, retorna os materiais mais semelhantes com base na pontuação de similaridade.
2. Detecção de duplicados — Compara toda a base de materiais para encontrar possíveis materiais duplicados de acordo com descrição, família e unidade de medida.

# Estrutura
- `main.py` → API em FastAPI para sugerir materiais semelhantes e registrar feedback dos usuários.
- `pesquisa_por_similaridade/` → busca por materiais parecidos
- `pesquisa_duplicados/` → varredura geral para encontrar materiais duplicados
- `requirements.txt` → bibliotecas necessárias

# Endpoints
- POST /buscar → Recebe descrição, UM e família e retorna top N semelhantes.
- POST /feedback → Registra se a sugestão foi correta ou não.

# Instalação
```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload