from AlgoritmoGenetico import AlgoritmoGenetico
from math import sqrt, sin


funcaoBase = lambda x,y: sin(x) + sqrt(x) - (y/3)
algGen = AlgoritmoGenetico(tamanho_populacao=10, funcaoBase=funcaoBase, maximoGeracoes=50)

algGen.gerarPopulacaoInicial()
algGen.selecaoRoleta()
algGen.plotarGrafico3d()
algGen.plotarGrafico2d()
