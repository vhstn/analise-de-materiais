import pandas as pd
import unicodedata
import recordlinkage
import time

def remover_acentos(texto):
    if pd.isna(texto):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFKD', str(texto)) if not unicodedata.combining(c))

def encontrar_duplicatas_recordlinkage_v2(dados, limiar=0.95, bonus_um=0.05, window=9):
    inicio = time.time()
    df = dados.copy().reset_index(drop=True)
    df['DESCRICAO_NORM'] = df['DESCRICAO'].apply(lambda x: remover_acentos(x).upper())

    indexer = recordlinkage.Index()
    indexer.sortedneighbourhood(left_on='DESCRICAO_NORM', window=window)
    candidatos = indexer.index(df)

    comp = recordlinkage.Compare()
    comp.string('DESCRICAO_NORM', 'DESCRICAO_NORM', method='jarowinkler', label='sim_desc')
    comp.exact('UM', 'UM', label='mesma_um')
    resultados = comp.compute(candidatos, df)

    resultados['score_final'] = resultados['sim_desc'] + (resultados['mesma_um'] * bonus_um)
    pares = resultados[resultados['score_final'] >= limiar].reset_index()
    pares = pares[pares['level_0'] != pares['level_1']]

    if pares.empty:
        fim = time.time()
        print(f"Tempo: {fim - inicio:.2f}s — nenhum par acima do limiar {limiar}")
        return pd.DataFrame(columns=[
            'CODIGO_1','DESCRICAO_1','UM_1','CODIGO_2','DESCRICAO_2','UM_2','score_final'
        ])

    pares['CODIGO_1'] = pares['level_0'].apply(lambda i: df.at[i, 'CODIGO'])
    pares['DESCRICAO_1'] = pares['level_0'].apply(lambda i: df.at[i, 'DESCRICAO'])
    pares['UM_1'] = pares['level_0'].apply(lambda i: df.at[i, 'UM'])
    pares['CODIGO_2'] = pares['level_1'].apply(lambda i: df.at[i, 'CODIGO'])
    pares['DESCRICAO_2'] = pares['level_1'].apply(lambda i: df.at[i, 'DESCRICAO'])
    pares['UM_2'] = pares['level_1'].apply(lambda i: df.at[i, 'UM'])

    pares = pares[pares['CODIGO_1'] != pares['CODIGO_2']]
    pares['par_unico'] = pares.apply(lambda r: tuple(sorted((r['CODIGO_1'], r['CODIGO_2']))), axis=1)
    pares = pares.drop_duplicates(subset='par_unico').drop(columns='par_unico')

    out = pares[['CODIGO_1','DESCRICAO_1','UM_1','CODIGO_2','DESCRICAO_2','UM_2','score_final']].sort_values(by='score_final', ascending=False)

    fim = time.time()
    print(f"Tempo: {fim - inicio:.2f}s — {len(out)} pares encontrados")
    return out
