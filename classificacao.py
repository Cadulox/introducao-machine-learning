# ALGORITMO SIMPLES DE CLASSIFICAÇÃO (PORCO OU CACHORRO)
from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()

# [é gordinho?, tem perninha curta?, faz auau? 0 = não e 1 = sim]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro4 = [1, 1, 1]
cachorro5 = [0, 1, 1]
cachorro6 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro4, cachorro5, cachorro6]

# um positivo(1) = porco / um negativo(-1) = cachorro
marcacoes = [1, 1, 1, -1, -1, -1]

# Treinar o modelo
modelo.fit(dados, marcacoes)

# Dataset simples de exemplo
misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

teste = [misterioso1, misterioso2, misterioso3]

# Suposto resultado esperado de saída
marcacoes_teste = [-1, 1, -1]
# marcacoes_teste = [-1, 1, 1]

# Prever o novo conjunto de dados
resutlado = modelo.predict(teste)

# Se o resultado abaixo der 0 para os elementos, o algoritmo acertou
diferencas = resutlado - marcacoes_teste

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos =len(teste)

# Taxa de acerto
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(resutlado)
print(diferencas)
print(taxa_de_acerto)
