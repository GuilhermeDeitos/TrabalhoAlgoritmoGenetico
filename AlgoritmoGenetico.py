
from math import sqrt, sin
import random
import numpy as np
import pandas as pd

class AlgoritmoGenetico:
    #Codigifação Binaria - OK
    #Seleção por roleta  - Imperfeita
    #Cruzamento de 2 ponto aleatorios - OK
    #Mutação por inversão binaria - OK (Eu acho)
    #1% de elitismo - OK (Eu acho)
    def __init__(self, tamanho_populacao, funcaoBase, taxaMutacao = 0.03, taxaElitismo = 0.01):
        self.tamanho_populacao:int = tamanho_populacao
        self.populacao:list = []
        self.geracao:int = 0
        self.melhor_solucao:dict = None
        self.melhor_solucao_geracao:int = 0
        self.funcaoBase = funcaoBase
        self.taxaMutacao = taxaMutacao
        self.taxaElitismo = taxaElitismo
       
    
    def converteBinToFloat(self, binario):
        return int(binario, 2)
    
    def converteFloatToBin(self, numero):
        return bin(numero)
    
    
    def calcFitness(self, x, y):
        fitness = self.funcaoBase(x, y)
        if fitness < 0:
            return abs(fitness)
        return fitness
        
    
    def gerarPopulacaoInicial(self):
        for i in range(self.tamanho_populacao):
            x = random.randint(0, 15)
            y = random.randint(0, 10)
            individuo = {}
            individuo["x"] = self.converteFloatToBin(x)
            individuo["y"] = self.converteFloatToBin(y)
            individuo["fitness"] = self.calcFitness(x,y) #linha a ser corrigida, pois interfere na seleção
            self.populacao.append(individuo)
      
        print("População Original: ")
        self.printIndividuos(self.populacao)
            
    def selecaoRoleta(self): #Meotodo da roleta de seleção
        somaFitness = sum([i["fitness"] for i in self.populacao])
        for i in self.populacao:
            i["porcentagem"] = i["fitness"]/somaFitness

        #Agora selecionar para realizar o cruzamento
        individuos = self.populacao
        novaPopulacao = []
        for i in range(self.tamanho_populacao):
            isElitismo = random.random() < self.taxaElitismo
            if isElitismo:
                individuo = max(individuos, key=lambda x:x["fitness"])
                novaPopulacao.append(individuo)
                individuos.remove(individuo)
            else:
                if(len(individuos) == 0):
                    break
                individuo1 = random.choices(individuos, weights=[i["porcentagem"] for i in individuos])[0]
                individuo2 = random.choices(individuos, weights=[i["porcentagem"] for i in individuos])[0]
                while(individuo1 == individuo2):
                    individuo2 = random.choices(individuos, weights=[i["porcentagem"] for i in individuos])[0]
                    
                filho1, filho2 = self.cruzamento(individuo1, individuo2)
                novaPopulacao.append(self.mutacao(filho1))
                novaPopulacao.append(self.mutacao(filho2))
                individuos.remove(individuo1)
                individuos.remove(individuo2)
                
        self.populacao = novaPopulacao
        self.geracao += 1
        print(f"População Após cruzamento da {self.geracao} geração: ")
        self.printIndividuos(novaPopulacao)
        
        
    
    def mutacao(self, individuo): #Mutação por inversão binaria
        if random.random() < self.taxaMutacao:
            individuoMutado = self.inversaoBinaria(individuo)
            print("Mutação")
            return individuoMutado
        return individuo
        
    def inversaoBinaria(self, individuo):
        individuo["x"] = individuo["x"][:random.randint(0, len(individuo["x"]))] + "1" + individuo["x"][random.randint(0, len(individuo["x"])):]
        individuo["y"] = individuo["y"][:random.randint(0, len(individuo["y"]))] + "1" + individuo["y"][random.randint(0, len(individuo["y"])):]
        individuo["fitness"] = self.calcFitness(self.converteBinToFloat(individuo["x"]), self.converteBinToFloat(individuo["y"]))
        return individuo
     
                
            
                   
        
    def cruzamento(self, individuo1:dict, individuo2:dict): #Cruzamento por 2 pontos aleatorios
        ponto1 = random.randint(0, len(individuo1["x"]))
        ponto2 = random.randint(ponto1, len(individuo1["x"]))
        filho1 = {}
        filho2 = {}
        filho1["x"] = individuo1["x"][:ponto1] + individuo2["x"][ponto1:ponto2] + individuo1["x"][ponto2:]
        filho1["y"] = individuo1["y"][:ponto1] + individuo2["y"][ponto1:ponto2] + individuo1["y"][ponto2:]
        filho2["x"] = individuo2["x"][:ponto1] + individuo1["x"][ponto1:ponto2] + individuo2["x"][ponto2:]
        filho2["y"] = individuo2["y"][:ponto1] + individuo1["y"][ponto1:ponto2] + individuo2["y"][ponto2:]
        
        filho1["fitness"] = self.calcFitness(self.converteBinToFloat(filho1["x"]),self.converteBinToFloat(filho1["y"]))
        filho2["fitness"] = self.calcFitness(self.converteBinToFloat(filho2["x"]),self.converteBinToFloat(filho2["y"]))
        return filho1, filho2
    
    def printIndividuos(self, individuos:list):
        df = pd.DataFrame(individuos)
        print(df)

        