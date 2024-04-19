from AlgoritmoGenetico import AlgoritmoGenetico
from math import sqrt, sin


funcaoBase = lambda x,y: sin(x) + sqrt(x) - (y/3)
algGen = AlgoritmoGenetico(tamanho_populacao=10, funcaoBase=funcaoBase, maximoGeracoes=100)

algGen.gerarPopulacaoInicial()
algGen.selecaoRoleta()
algGen.exibirMelhorSolucao()
