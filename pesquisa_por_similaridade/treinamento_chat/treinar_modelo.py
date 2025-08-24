import spacy
import random
import logging
from pathlib import Path
from spacy.training import Example

# Configurações de log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definir o diretório de saída para salvar o modelo
output_dir = Path("./pesquisa_por_similaridade/treinamento_chat/treinamento_chat_materiais")

def treinar_ner(n_iter=300):
    """
    Treina o modelo de reconhecimento de entidades com os dados fornecidos.
    """
    logging.info("Iniciando o processo de treinamento do modelo NER.")

    # Carregar os dados de treino
    try:
        from dados_treino import dados_treino
        if not dados_treino:
            logging.error("O arquivo não contém dados. Abortando.")
            return
        logging.info(f"Dados de treino carregados com sucesso. Total de exemplos: {len(dados_treino)}")
    except ImportError:
        logging.error("Erro: O arquivo não foi encontrado.")
        return
    except Exception as e:
        logging.error(f"Erro inesperado ao carregar os dados de treino: {e}")
        return

    # Inicializar o pipeline do spaCy
    try:
        nlp = spacy.blank("pt")
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        
        logging.info("Componente NER adicionado ao pipeline do spaCy.")

        # Adicionar os rótulos de entidades
        for _, annotations in dados_treino:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

    except Exception as e:
        logging.error(f"Erro ao inicializar o pipeline do spaCy ou adicionar rótulos: {e}")
        return

    # Treinar o modelo
    try:
        # Converter os dados de treino para o novo formato (objetos Example)
        examples = []
        for text, annotations in dados_treino:
            examples.append(Example.from_dict(nlp.make_doc(text), annotations))

        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            for i in range(n_iter):
                random.shuffle(examples)
                losses = {}
                nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)
                logging.info(f"Iteração {i+1}/{n_iter} - Perda: {losses['ner']:.4f}")
    
    except Exception as e:
        logging.error(f"Erro durante o treinamento: {e}")
        logging.error("Verifique a formatação dos seus dados de treino. A lista de exemplos precisa ser '[(texto, {'entities': [(...)]}), ...]'")
        return

    # Salvar o modelo treinado
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(output_dir)
        logging.info(f"Modelo spaCy treinado e salvo com sucesso em: {output_dir}")
    except Exception as e:
        logging.error(f"Erro ao salvar o modelo treinado: {e}")

if __name__ == '__main__':
    treinar_ner()