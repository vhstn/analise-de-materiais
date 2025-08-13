import pandas as pd
from buscar_parecidos import buscar_parecidos_manual

df1 = pd.read_csv("materiais_1.csv", encoding="ISO-8859-1", sep=";", on_bad_lines="skip")
df2 = pd.read_csv("materiais_2.csv", encoding="ISO-8859-1", sep=";", on_bad_lines="skip")
dados = pd.concat([df1, df2], ignore_index=True)[['CODIGO', 'DESCRICAO', 'UM', 'FAMILIA']]

resultados = buscar_parecidos_manual(
    descricao="PRENSA CABO",
    um="PC",
    familia=303,
    dados=dados,
    top_n=50
)

print(resultados)
