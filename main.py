from AlgoritmoGenetico import AlgoritmoGenetico
from math import sqrt, sin


funcaoBase = lambda x,y: sin(x) + sqrt(x) - (y/3)

try:
    sizePop = int(input("Informe o tamanho da população: "))
    maxGen = int(input("Maximo de gerações: "))
except ValueError:
    print("Por favor, informe um número inteiro.")
    exit()
algGen = AlgoritmoGenetico(tamanho_populacao=sizePop, funcaoBase=funcaoBase, maximoGeracoes=maxGen)

algGen.gerarPopulacaoInicial()
algGen.selecaoRoleta()
algGen.exibirMelhorSolucao()
