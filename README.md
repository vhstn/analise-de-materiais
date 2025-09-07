# Plataforma de Análise de Materiais com IA e Microserviços

Este projeto é uma plataforma completa para busca e análise de materiais, construída com uma arquitetura de microserviços. O sistema utiliza um modelo de Inteligência Artificial (Reconhecimento de Entidades Nomeadas - NER) para extrair informações de texto em linguagem natural e possui um ciclo de feedback que permite o retreinamento contínuo do modelo.

A aplicação é totalmente containerizada com Docker, garantindo portabilidade e facilidade de implantação.

## Funcionalidades Principais

  * **Busca por Chat (NER):** Permite que o usuário busque materiais através de uma interface de chat, usando linguagem natural. O modelo, treinado com **spaCy**, extrai as entidades `DESCRICAO`, `UM` e `FAMILIA` para realizar a busca.
  * **Loop de Feedback Contínuo:** Uma interface interativa permite que o usuário corrija erros de extração do modelo. O feedback enviado dispara um processo de retreinamento assíncrono, melhorando a precisão do modelo com o uso.
  * **Processamento Assíncrono:** O retreinamento do modelo, uma tarefa computacionalmente intensiva, é delegado a um worker assíncrono usando **Celery** e **RabbitMQ**, garantindo que a API principal permaneça rápida e responsiva.
  * **API Segura e Documentada:** A comunicação é protegida por token de autenticação, e a API é totalmente documentada via Swagger UI.

## Arquitetura de Microserviços

O sistema é orquestrado pelo Docker Compose e dividido em quatro serviços principais:

1.  **API (FastAPI):** O cérebro da aplicação. Recebe as requisições, utiliza o modelo para fazer predições e envia tarefas de retreinamento.
2.  **Interface (Streamlit):** O frontend da aplicação, onde o usuário interage com o chat e fornece feedbacks.
3.  **Message Broker (RabbitMQ):** A fila de mensagens que desacopla a API do worker. Garante que as tarefas de retreinamento não sejam perdidas e sejam processadas de forma ordenada.
4.  **Worker (Celery):** O serviço que consome as tarefas da fila do RabbitMQ e executa o processo pesado de retreinamento do modelo.

## Tecnologias Utilizadas

  * **Backend:** Python, FastAPI
  * **Frontend:** Streamlit
  * **Machine Learning (NER):** spaCy
  * **Mensageria e Tarefas Assíncronas:** RabbitMQ, Celery
  * **Containerização:** Docker, Docker Compose
  * **Banco de Dados (Busca):** Pandas (lendo de CSV)
  * **Segurança:** python-dotenv, python-jose

## Executando o Projeto

A forma recomendada de executar esta aplicação é com Docker. Ela possui uma planilha com materiais de exemplo (materiais.csv).

### 1\. Executando com Docker (Recomendado)

Esta abordagem executa toda a aplicação em contêineres isolados.

1.  **Instale o Docker:**
    O único pré-requisito é ter o **Docker Desktop** (que inclui o Docker Compose) instalado na sua máquina.

    * [Download do Docker para Windows/Mac](https://www.docker.com/products/docker-desktop/)

2.  **Clone o repositório:**

    ```bash
    git clone https://github.com/vhstn/analise-de-materiais.git
    cd analise-de-materiais
    ```

3.  **Crie o arquivo de ambiente:**
    Crie um arquivo chamado .env na raiz do projeto. Este arquivo guardará todas as suas credenciais de forma segura. Adicione as seguintes variáveis, substituindo pelos seus valores:
    ```env
    # Chave de acesso para a API
    API_KEY="seu_token"

    # Credenciais para o RabbitMQ
    RABBITMQ_USER="seu_usuario"
    RABBITMQ_PASS="sua_senha"
    ```

4.  **Inicie a aplicação:**
    Execute o seguinte comando na raiz do projeto. Ele irá construir as imagens, baixar o RabbitMQ e iniciar todos os serviços.

    ```bash
    docker-compose up --build
    ```

5.  **Acesse os serviços:**

    * **Interface Streamlit:** `http://localhost:8501`
    * **Documentação da API (Swagger):** `http://localhost:8000/docs`
    * **Painel de Gerenciamento do RabbitMQ:** `http://localhost:15672`

Para parar a aplicação, pressione `CTRL + C` no terminal.

### 2\. Executando Localmente (Para Desenvolvimento)

Esta abordagem requer que você tenha o Python e o RabbitMQ instalados na sua máquina.

1.  **Instale os pré-requisitos:**

    * Instale o Python 3.11+.
    * Instale e inicie o [RabbitMQ Server](https://www.rabbitmq.com/download.html).
    * Instale as dependências do projeto:
      ```bash
      pip install -r requirements.txt
      ```

2.  **Crie o arquivo de ambiente** (`.env`), como descrito no método Docker.

3.  **Execute o treinamento inicial:**

    ```bash
    python pesquisa_por_similaridade/treinamento_chat/treinar_modelo.py
    ```

4.  **Inicie os serviços (em 3 terminais separados):**

    * **Terminal 1 (API):**
      ```bash
      python -m uvicorn pesquisa_por_similaridade.main:app --reload
      ```
    * **Terminal 2 (Worker Celery):**
      ```bash
      python -m celery -A pesquisa_por_similaridade.celery_worker worker --loglevel=info
      ```
    * **Terminal 3 (Interface):**
      ```bash
      python -m streamlit run app_interface.py
      ```