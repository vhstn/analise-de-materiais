# Análise de materiais

Este projeto contém duas funções para análise de materiais:

1. Sugestão de materiais mais parecidos — Dada uma descrição, unidade de medida e família, retorna os materiais mais semelhantes com base na pontuação de similaridade.
2. Detecção de duplicados — Compara toda a base de materiais para encontrar possíveis materiais duplicados de acordo com descrição, família e unidade de medida.

# Estrutura
- `main.py` → API em FastAPI para sugerir materiais semelhantes e registrar feedback dos usuários.
- `pesquisa_por_similaridade/` → busca por materiais parecidos
- `pesquisa_duplicados/` → varredura geral para encontrar materiais duplicados
- `dados_treino/` → frases para contextualizar a spaCy e identificar as entidades da frase.
- `treinar_modelo/` → treina o modelo com base nos dados.
- `requirements.txt` → bibliotecas necessárias

# Endpoints
- POST /buscar → Recebe descrição, UM e família e retorna top N semelhantes.
- POST /chat → Carrega uma mensagem e identifica as entidades: descrição, unidade de medida e família do material. Após isso, apresenta as sugestões.
- POST /feedback-ner → Recebe correções de entidades, formata como dados de treino para o spaCy e inicia o processo de re-treinamento.

# Instalação
```bash
pip install -r requirements.txt
python pesquisa_por_similaridade/treinamento_chat/treinar_modelo.py
python -m uvicorn pesquisa_por_similaridade.main:app --reload
