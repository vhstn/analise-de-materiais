import pandas as pd
from encontrar_duplicatas import encontrar_duplicatas_recordlinkage_v2

df1 = pd.read_csv("materiais_1.csv", encoding="ISO-8859-1", sep=";", on_bad_lines="skip")
df2 = pd.read_csv("materiais_2.csv", encoding="ISO-8859-1", sep=";", on_bad_lines="skip")
dados = pd.concat([df1, df2], ignore_index=True)[['CODIGO', 'DESCRICAO', 'UM', 'FAMILIA']]

df_duplicatas = encontrar_duplicatas_recordlinkage_v2(dados, limiar=1, bonus_um=0.05, window=9)
df_duplicatas.to_excel("duplicatas_detectadas.xlsx", index=False)

print(df_duplicatas)
