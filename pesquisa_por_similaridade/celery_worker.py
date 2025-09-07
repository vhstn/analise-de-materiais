import os
import logging
from celery import Celery

# Configura a conexão com o RabbitMQ usando uma variável de ambiente
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")

# Cria a instância principal do Celery
celery_app = Celery("tasks", broker=BROKER_URL)

# Configura o logging para o worker
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@celery_app.task(name="retreinar_modelo_task")
def retreinar_modelo_task():
    """
    Tarefa do Celery que executa a função de retreinamento.
    A importação é feita DENTRO da função para evitar o circular import.
    """
    try:
        # Importações feitas em tempo de execução
        from .main import model_manager
        from .retreinar_com_feedback import retreinar_modelo_ner

        logging.info("Worker Celery: Tarefa de retreinamento recebida. Iniciando processo.")
        retreinar_modelo_ner(model_manager)
        logging.info("Worker Celery: Processo de retreinamento concluído com sucesso.")
    except Exception as e:
        logging.error(f"Worker Celery: Erro durante o retreinamento: {e}")