import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

class AlgoritmoGenetico:
    def __init__(self, maximoGeracoes, tamanho_populacao, funcaoBase, taxaMutacao = 0.03, taxaElitismo = 0.01):
        self.tamanho_populacao:int = tamanho_populacao
        self.populacao:list = []
        self.geracao:int = 0
        self.melhor_solucao:dict = None
        self.melhor_solucao_geracao:int = 0
        self.maximoGeracoes:int = maximoGeracoes
        self.funcaoBase = funcaoBase
        self.taxaMutacao = taxaMutacao
        self.taxaElitismo = taxaElitismo
       
    
    def converteBinToFloat(self, binario):
        inteiro,dec = binario.split('.')        
        inteiro = int(inteiro,2)
        dec = int(dec, 2)
        result = inteiro + dec/(10**len(str(dec)))
        return result
    
    def converteFloatToBin(self, numero, casas=10):
        # Converte a parte inteira
        parte_inteira = int(numero)
        parte_inteira_bin = bin(parte_inteira).replace("0b", "")

        # Converte a parte decimal
        parte_decimal = numero - parte_inteira
        parte_decimal_bin = ""
        while casas:
            parte_decimal *= 2
            bit = int(parte_decimal)

            if bit == 1:
                parte_decimal -= bit
                parte_decimal_bin += '1'
            else:
                parte_decimal_bin += '0'

            casas -= 1

        # Combina as partes inteira e decimal
        numero_bin = parte_inteira_bin + "." + parte_decimal_bin
        print(f"Binario: {numero_bin} Decimal: {numero}")
        return numero_bin

    def decimal_converter(self,num): 
        while num > 1:
            num /= 10
        return num
        
    
    def calcFitness(self, x, y):
        fitness = self.funcaoBase(x, y)
        if fitness < 0:
            return abs(fitness)
        return fitness
        
    
    def gerarPopulacaoInicial(self):
        for i in range(self.tamanho_populacao):
            # Gerar um float aleatório x e y
            x = round(random.uniform(0, 15), 3)
            y = round(random.uniform(0, 10), 3)
            
            individuo = {}
            individuo["x"] = self.converteFloatToBin(x)
            individuo["y"] = self.converteFloatToBin(y)
            individuo["fitness"] = self.calcFitness(x,y)
            individuo["porcentagem"] = 0  # Inicializa a porcentagem da população
            self.populacao.append(individuo)
        
        # Calcular a soma do fitness da população completa
        somaFitness = sum([i["fitness"] for i in self.populacao])
        
        # Calcular as porcentagens para todos os indivíduos
        for individuo in self.populacao:
            individuo["porcentagem"] = individuo["fitness"] * 100 / (somaFitness + 1e-9) # Calcula a porcentagem inicial da população
        
        print("População Original: ")
        self.printIndividuos(self.populacao)
       
        
    def selecaoRoleta(self):
        fitness_medio_por_geracao = []
        while self.maximoGeracoes > self.geracao:
            individuos = self.populacao.copy()
            novaPopulacao = []
            fitness_total = 0
            
            for i in range(self.tamanho_populacao):
                isElitismo = random.random() < self.taxaElitismo
                if isElitismo:
                    if not individuos:
                        break
                    individuo = max(individuos, key=lambda x:x["fitness"])
                    if self.melhor_solucao is None or individuo["fitness"] > self.melhor_solucao["fitness"]:
                        self.melhor_solucao = individuo
                        self.melhor_solucao_geracao = self.geracao
                    novaPopulacao.append(individuo)
                    individuos.remove(individuo)
                else:
                    individuosRestantes = individuos.copy()
                    for _ in range(len(individuos) // 2):
                        individuo1 = random.choices(individuosRestantes, weights=[i["porcentagem"] for i in individuosRestantes])[0]
                        individuosRestantes.remove(individuo1)
                        individuo2 = random.choices(individuosRestantes, weights=[i["porcentagem"] for i in individuosRestantes])[0]
                        individuosRestantes.remove(individuo2)
                        filho1, filho2 = self.cruzamento(individuo1, individuo2)
                        novaPopulacao.append(self.mutacao(filho1))
                        novaPopulacao.append(self.mutacao(filho2))
            
            # Calcula o valor médio do fitness
            fitness_total = sum([individuo["fitness"] for individuo in novaPopulacao])
            fitness_medio_por_geracao.append(fitness_total / len(novaPopulacao))
            
            somaFitnessNovaPopulacao = sum([i["fitness"] for i in novaPopulacao])
            for individuo in novaPopulacao:
                individuo["porcentagem"] = individuo["fitness"] * 100 / (somaFitnessNovaPopulacao + 1e-9) # Calcula a porcentagem para cada novo individuo da população
                if self.melhor_solucao is None or individuo["fitness"] > self.melhor_solucao["fitness"]:
                    self.melhor_solucao = individuo
                    self.melhor_solucao_geracao = self.geracao
            self.geracao += 1      
            self.printIndividuos(novaPopulacao)
            
            self.plotarGrafico3d()
            self.plotarGrafico2d()
        self.plotarGraficoFitness(fitness_medio_por_geracao)
            
        self.populacao = novaPopulacao       
        print(f"População Após cruzamento da {self.geracao} geração: ")

        

    def mutacao(self, individuo): #Mutação por inversão binaria
        if random.random() < self.taxaMutacao:
            individuoMutado = self.inversaoBinaria(individuo)
            print("Mutação")
            return individuoMutado
        return individuo
        

    def inversaoBinaria(self, individuo):
        inteiro, dec = individuo["x"].split(".")
        inteiro, dec = self.auxInversao(list(inteiro)), self.auxInversao(list(dec))
        inteiro, dec = individuo["y"].split(".")
        inteiro, dec = self.auxInversao(list(inteiro)), self.auxInversao(list(dec))
        
        individuo["y"] = "".join(inteiro) + "." + "".join(dec)
        individuo["x"] = "".join(inteiro) + "." + "".join(dec)
        
        individuo["fitness"] = self.calcFitness(self.converteBinToFloat(individuo["x"]), self.converteBinToFloat(individuo["y"]))
        print("Cruzamento nulo ",individuo["fitness"] == 0)
        return individuo
     
    def auxInversao(self, listIndividuo):
        return ['1' if i == '0' else '0' for i in listIndividuo]
            
                   
    def cruzamento(self, individuo1:dict, individuo2:dict): #Cruzamento por 2 pontos aleatorios
        filho1 = {"x": "", "y": ""}
        filho2 = {"x": "", "y": ""}

        for coord in ["x", "y"]:
            parte_inteira1, parte_decimal1 = individuo1[coord].split('.')
            parte_inteira2, parte_decimal2 = individuo2[coord].split('.')

            ponto_corte_inteiro = random.randint(1, len(parte_inteira1) - 1) if len(parte_inteira1) > 1 else 0
            ponto_corte_decimal = random.randint(1, len(parte_decimal1) - 1) if len(parte_decimal1) > 1 else 0

            filho1[coord] = parte_inteira1[:ponto_corte_inteiro] + parte_inteira2[ponto_corte_inteiro:] + '.' + parte_decimal1[:ponto_corte_decimal] + parte_decimal2[ponto_corte_decimal:]
            filho2[coord] = parte_inteira2[:ponto_corte_inteiro] + parte_inteira1[ponto_corte_inteiro:] + '.' + parte_decimal2[:ponto_corte_decimal] + parte_decimal1[ponto_corte_decimal:]

        filho1["fitness"] = self.calcFitness(self.converteBinToFloat(filho1["x"]),self.converteBinToFloat(filho1["y"]))
        filho2["fitness"] = self.calcFitness(self.converteBinToFloat(filho2["x"]),self.converteBinToFloat(filho2["y"]))
        return filho1, filho2
    

    def exibirMelhorSolucao(self):
            print("Melhor solução encontrada:")
            print("X:", self.converteBinToFloat(self.melhor_solucao["x"]))
            print("Y:", self.converteBinToFloat(self.melhor_solucao["y"]))
            print("Fitness:", self.melhor_solucao["fitness"])
            print("Geração:", self.melhor_solucao_geracao)

    
    def getListOfDict(self,lista:list):
        numKeys = len(lista[0].keys())
        listaFinal = []
        for i in range(0,numKeys):
            listaAux = []
            for item in lista:
                listaAux.append(item[list(item.keys())[i]])
            listaFinal.append(listaAux)

        return listaFinal
    

    def plotarGrafico3d(self):
        populacaoFloat = []
        for i in self.populacao:
            if self.melhor_solucao == None or i["fitness"] > self.melhor_solucao["fitness"]:
                self.melhor_solucao = i
                self.melhor_solucao_geracao = self.geracao
            print(i)
            populacaoFloat.append({"x":self.converteBinToFloat(i["x"]), "y":self.converteBinToFloat(i["y"]), "fitness":i["fitness"]})
            
        print("População: ", populacaoFloat)
        
        plotIndList = self.getListOfDict(populacaoFloat)
        plt.style.use('_mpl-gallery')
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(np.array(plotIndList[0]), np.array(plotIndList[1]))
        calcFitness_vect = np.vectorize(self.calcFitness)
        Z = calcFitness_vect(X, Y)
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0.5)
        ax.set(xlabel="X", ylabel="Y", zlabel="Fitness")
        fig.set_size_inches(10, 10)

        plt.show()


    def plotarGrafico2d(self):
        populacaoFloat = []

        for i in self.populacao:
            populacaoFloat.append({"x":self.converteBinToFloat(i["x"]), "y":self.converteBinToFloat(i["y"]), "fitness":i["fitness"]})

        plotIndList = self.getListOfDict(populacaoFloat)

        X, Y = np.meshgrid(np.array(plotIndList[0]), np.array(plotIndList[1]))
        calcFitness_vect = np.vectorize(self.calcFitness)
        Z = calcFitness_vect(X, Y)

        plt.imshow(Z, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Fitness')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    
    def plotarGraficoFitness(self, fitness_medio_por_geracao):
        plt.plot(range(1, len(fitness_medio_por_geracao) + 1), fitness_medio_por_geracao)
        plt.title('Valor Médio do Fitness por Geração')
        plt.xlabel('Geração')
        plt.ylabel('Fitness Médio')
        plt.show()

        
    def printIndividuos(self, individuos:list):
        df = pd.DataFrame(individuos)
        print(df)
