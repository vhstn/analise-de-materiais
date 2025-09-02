# Imagem oficial do Python 3.11
FROM python:3.11-slim

# Diretório onde a aplicação ficará dentro do contêiner
WORKDIR /app

# Copia o arquivo de dependências primeiro para otimizar o cache do Docker
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o resto do código da sua aplicação para dentro do contêiner
COPY . .

# Cria um usuário não-root
RUN useradd -m appuser

# Muda para o novo usuário
USER appuser

# Informa ao Docker que a API usará a porta 8000 e a interface a 8501
EXPOSE 8000
EXPOSE 8501

CMD ["python", "-m", "uvicorn", "pesquisa_por_similaridade.main:app", "--host", "0.0.0.0", "--port", "8000"]