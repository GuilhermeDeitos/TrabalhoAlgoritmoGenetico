from AlgoritmoGenetico import AlgoritmoGenetico
from math import sqrt, sin


funcaoBase = lambda x,y: sin(x) + sqrt(x) - (y/3)
print(funcaoBase(15,0))
algGen = AlgoritmoGenetico(10, funcaoBase)

algGen.gerarPopulacaoInicial()
algGen.selecaoRoleta()
