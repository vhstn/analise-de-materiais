import os
import logging
import spacy
import json
from spacy.training import Example
import random
import pandas as pd

def retreinar_modelo_ner():
    """
    Carrega o modelo spaCy, lê os novos dados de feedback e atualiza o componente NER.
    """
    MODEL_PATH = "./treinamento_chat/treinamento_chat_materiais"
    FEEDBACK_NER_FILE = "./treinamento_chat/dados_aprendizado.jsonl"    

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