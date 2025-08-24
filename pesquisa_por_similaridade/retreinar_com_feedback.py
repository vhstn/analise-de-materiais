import os
import logging
import spacy
import json
from spacy.training import Example
import random
from pathlib import Path

def retreinar_modelo_ner(nlp_instance):
    """
    Carrega o modelo spaCy, lê os novos dados de feedback e atualiza o componente NER.
    """
    MODEL_PATH = "./pesquisa_por_similaridade/treinamento_chat/treinamento_chat_materiais"
    FEEDBACK_NER_FILE = "./pesquisa_por_similaridade/treinamento_chat/novos_dados_treino.jsonl"

    if not os.path.exists(FEEDBACK_NER_FILE):
        logging.info("Nenhum novo dado de feedback para treinar.")
        return

    try:
        logging.info("Iniciando o processo de re-treinamento do modelo NER.")
        
        # Carrega os novos dados de treino do arquivo de feedback
        train_examples = []
        with open(FEEDBACK_NER_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data[0]
                annotations = data[1]
                train_examples.append(Example.from_dict(nlp_instance.make_doc(text), annotations))

        if not train_examples:
            logging.info("Arquivo de feedback encontrado, mas está vazio. Nenhum treinamento necessário.")
            return

        logging.info(f"Re-treinando com {len(train_examples)} novos exemplos.")

        # Desativa outros pipes para focar no NER
        other_pipes = [pipe for pipe in nlp_instance.pipe_names if pipe != "ner"]
        
        with nlp_instance.disable_pipes(*other_pipes):
            optimizer = nlp_instance.resume_training()
            for i in range(10):
                random.shuffle(train_examples)
                losses = {}
                nlp_instance.update(train_examples, drop=0.35, losses=losses, sgd=optimizer)
                logging.info(f"Iteração de re-treino {i+1}/10 - Perda: {losses.get('ner', 0.0):.4f}")

        # Salva o modelo atualizado
        Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
        nlp_instance.to_disk(MODEL_PATH)
        logging.info(f"Modelo re-treinado e salvo com sucesso em: {MODEL_PATH}")

        # Limpa o arquivo de feedback após o uso bem-sucedido
        with open(FEEDBACK_NER_FILE, 'w') as f:
            f.write('')
        logging.info("Arquivo de feedback limpo para novos dados.")

    except Exception as e:
        logging.error(f"Erro durante o re-treinamento: {e}")