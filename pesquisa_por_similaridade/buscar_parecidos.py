import pandas as pd
from rapidfuzz import fuzz
import unicodedata

def remover_acentos(texto):
    if pd.isna(texto):
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', str(texto))
        if not unicodedata.combining(c)
    )

def buscar_parecidos_manual(descricao, um, familia, dados, top_n=5):
    # Normaliza a descrição de entrada
    descricao_norm = remover_acentos(descricao).upper()

    base = {
        'DESCRICAO': descricao_norm,
        'UM': um,
        'FAMILIA': familia
    }

    resultados = []

    for _, row in dados.iterrows():
        # Normaliza candidato
        desc_cand_norm = remover_acentos(row['DESCRICAO']).upper()

        # Similaridade textual
        sim_desc = fuzz.token_set_ratio(base['DESCRICAO'], desc_cand_norm)

        # Bônus por palavras exatas
        palavras_base = set(base['DESCRICAO'].split())
        palavras_cand = set(desc_cand_norm.split())
        match_exatas = len(palavras_base & palavras_cand) * 15

        # Bônus por família e UM
        bonus = 0
        if row['FAMILIA'] == base['FAMILIA']:
            bonus += 30
        if row['UM'] == base['UM']:
            bonus += 20

        score = sim_desc + bonus + match_exatas
        resultados.append((row['CODIGO'], row['DESCRICAO'], row['UM'], row['FAMILIA'], score))

    df_res = pd.DataFrame(resultados, columns=['CODIGO', 'DESCRICAO', 'UM', 'FAMILIA', 'SCORE'])
    return df_res.sort_values(by='SCORE', ascending=False).head(top_n)
