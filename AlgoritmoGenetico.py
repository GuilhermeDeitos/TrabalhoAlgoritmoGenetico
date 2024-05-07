import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

class AlgoritmoGenetico:
    #Construtor para inicializar algumas variaveis
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
        inteiro,dec = binario.split('.')  #Separa a parte inteira da parte decimal para a conversão   
        inteiro = int(inteiro,2)
        dec = int(dec, 2)
        result = inteiro + dec/(10**len(str(dec))) # Soma a parte inteira convertida com a parte decimal
        return result
    
    #Limitei para 10 casas decimais, porem poderia ser mais caso necessário
    def converteFloatToBin(self, numero, casas=10):
        # Converte a parte inteira
        parte_inteira = int(numero)
        parte_inteira_bin = bin(parte_inteira).replace("0b", "") #Como só trabalharemos com numeros inteiros, retiramos o 0b

        # Converte a parte decimal
        parte_decimal = numero - parte_inteira #Deixando só a parte decimal
        parte_decimal_bin = "" #Variavel auxiliar string para armazenar a parte decimal em binario
        
        while casas: #Enquanto não bater o numero de casas 
            parte_decimal *= 2 # Multiplica a parte decimal por 2
            bit = int(parte_decimal) #Converter a parte decimal para inteiro

            if bit == 1: #Se o bit for 1, subtrai 1 da parte decimal e adiciona 1 na parte decimal binaria
                parte_decimal -= bit
                parte_decimal_bin += '1'
            else:
                parte_decimal_bin += '0' #Se o bit for 0, adiciona 0 na parte decimal binaria

            casas -= 1 #Atualiza a variavel das casas

        # Combina as partes inteira e decimal
        numero_bin = parte_inteira_bin + "." + parte_decimal_bin
       # print(f"Binario: {numero_bin} Decimal: {numero}") #Print para debug
        return numero_bin        
    
    def calcFitness(self, x, y): #Calcula o fitness da função
        fitness = self.funcaoBase(x, y)
        if fitness < 0:
            return abs(fitness)
        return fitness
        
    
    def gerarPopulacaoInicial(self): 
        for i in range(self.tamanho_populacao): #Gera os individuos iniciais a partir do tamanho da população
            # Gerar um float aleatório x e y (Limitado para 3 casas decimais)
            x = round(random.uniform(0, 15), 3)
            y = round(random.uniform(0, 10), 3)
            
            individuo = {} #Cria um dicionario para armazenar os valores
            individuo["x"] = self.converteFloatToBin(x)
            individuo["y"] = self.converteFloatToBin(y)
            individuo["fitness"] = self.calcFitness(x,y)
            individuo["porcentagem"] = 0  # Inicializa a porcentagem da população
            self.populacao.append(individuo) #Adiciona o individuo na população
        
        # Calcular a soma do fitness da população completa
        somaFitness = sum([i["fitness"] for i in self.populacao])
        
        # Calcular as porcentagens para todos os indivíduos
        for individuo in self.populacao:
            individuo["porcentagem"] = individuo["fitness"] * 100 / (somaFitness + 1e-9) # Calcula a porcentagem inicial da população
        
        print("População Original: ")
        self.printIndividuos(self.populacao)
       
        
    def selecaoRoleta(self): #Metodo de seleção por roleta (maior fitness = maior chance de ser escolhido)
        fitness_medio_por_geracao = [] 
        while self.maximoGeracoes > self.geracao:  #Enquanto não atingir o numero maximo de gerações
            individuos = self.populacao
            novaPopulacao = [] #Cria uma nova população
            fitness_total = 0 #Inicializa a variavel fitness_total
            
            for i in range(self.tamanho_populacao): #Para cada individuo na população
                isElitismo = random.random() < self.taxaElitismo #Utilizado para permitir que o melhor individuo da população anterior passe para a próxima geração
                if isElitismo:
                    print("Elitismo") #Print para debug
                    if not individuos: 
                        break
                    individuo = max(individuos, key=lambda x:x["fitness"]) #Seleciona o melhor individuo da população anterior
                    if self.melhor_solucao is None or individuo["fitness"] > self.melhor_solucao["fitness"]: #Se o individuo for melhor que a melhor solução atual, atualiza a melhor solução
                        self.melhor_solucao = individuo
                        self.melhor_solucao_geracao = self.geracao 
                    novaPopulacao.append(individuo) #Adiciona o individuo na nova população
                    individuos.remove(individuo) #Remove o individuo da lista de individuos restantes
                else:
                    individuosRestantes = individuos 
                    for _ in range(len(individuos) // 2):   #Para cada individuo restante
                        individuo1 = random.choices(individuosRestantes, weights=[i["porcentagem"] for i in individuosRestantes])[0] #Escolhe um individuo aleatorio da lista de individuos restantes
                        individuosRestantes.remove(individuo1)
                        individuo2 = random.choices(individuosRestantes, weights=[i["porcentagem"] for i in individuosRestantes])[0]
                        individuosRestantes.remove(individuo2)
                        filho1, filho2 = self.cruzamento(individuo1, individuo2) #Realiza o cruzamento dos individuos

                        novaPopulacao.append(self.mutacao(filho1)) #Adiciona os filhos na nova população
                        novaPopulacao.append(self.mutacao(filho2))
                        
            
            # Calcula o valor médio do fitness
            fitness_total = sum([individuo["fitness"] for individuo in novaPopulacao])
            fitness_medio_por_geracao.append(fitness_total / len(novaPopulacao)) #Adiciona o valor médio do fitness na lista de fitness por geração
            
            somaFitnessNovaPopulacao = sum([i["fitness"] for i in novaPopulacao])  
            for individuo in novaPopulacao:
                individuo["porcentagem"] = individuo["fitness"] * 100 / (somaFitnessNovaPopulacao + 1e-9) # Calcula a porcentagem para cada novo individuo da população
                if self.melhor_solucao is None or individuo["fitness"] > self.melhor_solucao["fitness"]: #Verifica se é a melhor solução novamente
                    self.melhor_solucao = individuo
                    self.melhor_solucao_geracao = self.geracao
            self.geracao += 1      
            self.populacao = novaPopulacao  #Atualiza a população      

            #self.plotarGrafico3d() 
            #self.plotarGrafico2d() 
            #Mostrar população
            print("População: ")
            self.printIndividuos(self.populacao)
            
            
        self.plotarGraficoFitness(fitness_medio_por_geracao)
            

        

    def mutacao(self, individuo): #Mutação por inversão binaria
        if random.random() < self.taxaMutacao: #Verifica se o numero aleatório gerado é menor que a taxa de mutação
            individuoMutado = self.inversaoBinaria(individuo) #Realiza a mutação
            print("Mutação")  #Print para debug
            return individuoMutado
        return individuo 
        

    def inversaoBinaria(self, individuo): # Metodo da inversão binaria
        inteiro, dec = individuo["x"].split(".") #Separa a parte inteira da parte decimal
        inteiro, dec = self.auxInversao(list(inteiro)), self.auxInversao(list(dec)) #Inverte a parte inteira e a parte decimal do genoma x
        inteiro, dec = individuo["y"].split(".")
        inteiro, dec = self.auxInversao(list(inteiro)), self.auxInversao(list(dec)) #Inverte a parte inteira e a parte decimal do genoma y
        
        individuo["y"] = "".join(inteiro) + "." + "".join(dec) #Junta a parte inteira e a parte decimal do genoma y apos inversão
        individuo["x"] = "".join(inteiro) + "." + "".join(dec) #Junta a parte inteira e a parte decimal do genoma x apos inversão
        
        individuo["fitness"] = self.calcFitness(self.converteBinToFloat(individuo["x"]), self.converteBinToFloat(individuo["y"])) #Calcula o fitness do individuo após a mutação
        #print("Cruzamento nulo ",individuo["fitness"] == 0) #Print para debug
        return individuo
     
    def auxInversao(self, listIndividuo): #Metodo auxiliar para inverter o genoma
        return ['1' if i == '0' else '0' for i in listIndividuo] #Caso genoma[i] == 0, genoma[i] = 1, caso contrario genoma[i] = 0
            
                   
    def cruzamento(self, individuo1:dict, individuo2:dict): #Cruzamento por 2 pontos aleatorios
        #Inicialização dos filhyos
        filho1 = {"x": "", "y": ""}
        filho2 = {"x": "", "y": ""}

        for coord in ["x", "y"]: #Para cada genoma
            #Separa a parte inteira e a parte decimal do individuo 1 e 2
            parte_inteira1, parte_decimal1 = individuo1[coord].split('.') 
            parte_inteira2, parte_decimal2 = individuo2[coord].split('.') 

            ponto_corte_inteiro = random.randint(1, len(parte_inteira1) - 1) if len(parte_inteira1) > 1 else 0 #Gera um ponto de corte aleatorio para a parte inteira
            ponto_corte_decimal = random.randint(1, len(parte_decimal1) - 1) if len(parte_decimal1) > 1 else 0 #Gera um ponto de corte aleatorio para a parte decimal

            #Atualiza os filhos no genoma atual
            filho1[coord] = parte_inteira1[:ponto_corte_inteiro] + parte_inteira2[ponto_corte_inteiro:] + '.' + parte_decimal1[:ponto_corte_decimal] + parte_decimal2[ponto_corte_decimal:]  
            filho2[coord] = parte_inteira2[:ponto_corte_inteiro] + parte_inteira1[ponto_corte_inteiro:] + '.' + parte_decimal2[:ponto_corte_decimal] + parte_decimal1[ponto_corte_decimal:]

        #Calcula o fitness dos filhos
        filho1["fitness"] = self.calcFitness(self.converteBinToFloat(filho1["x"]),self.converteBinToFloat(filho1["y"]))
        filho2["fitness"] = self.calcFitness(self.converteBinToFloat(filho2["x"]),self.converteBinToFloat(filho2["y"]))
        return filho1, filho2
    

    def exibirMelhorSolucao(self): #Exibe a melhor solução encontrada
            print("Melhor solução encontrada:")
            print("X:", self.converteBinToFloat(self.melhor_solucao["x"]))
            print("Y:", self.converteBinToFloat(self.melhor_solucao["y"]))
            print("Fitness:", self.melhor_solucao["fitness"])
            print("Geração:", self.melhor_solucao_geracao)


    def getListOfDict(self,lista:list): #Função auxiliar para transformar a lista de dicionarios em uma lista de listas
        numKeys = len(lista[0].keys()) #Pega o numero de chaves do dicionario contido na lista
        listaFinal = []
        for i in range(0,numKeys): #Para cada chave do dicionario
            listaAux = []
            for item in lista: #Para cada item na lista
                listaAux.append(item[list(item.keys())[i]]) #Adiciona o valor da chave i na lista auxiliar
            listaFinal.append(listaAux) #Adiciona a lista auxiliar na lista final

        return listaFinal
    

    def plotarGrafico3d(self):
        populacaoFloat = [] #Lista para armazenar os individuos em float
        for i in self.populacao:             
            populacaoFloat.append({"x":self.converteBinToFloat(i["x"]), "y":self.converteBinToFloat(i["y"]), "fitness":i["fitness"]}) #Adiciona o individuo na lista de individuos em float
            
        #print("População: ", populacaoFloat) #Print para debug
        
        plotIndList = self.getListOfDict(populacaoFloat) #Transforma a lista de dicionarios em uma lista de listas (Para manipular na hora de plotar o grafico)
        plt.style.use('_mpl-gallery')
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) #Cria a figura e o eixo 3d
        X, Y = np.meshgrid(np.array(plotIndList[0]), np.array(plotIndList[1])) #Cria a malha de pontos para o grafico
        calcFitness_vect = np.vectorize(self.calcFitness) 
        Z = calcFitness_vect(X, Y) #Calcula o fitness para cada ponto da malha
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0.5) #Plota a superficie
        ax.set(xlabel="X", ylabel="Y", zlabel="Fitness") #Adiciona os labels
        plt.title(f"Fitness da população na geração {self.geracao}") 
        fig.set_size_inches(10, 10) #Define o tamanho da figura

        plt.show()


    def plotarGrafico2d(self):
        populacaoFloat = []

        for i in self.populacao:
            populacaoFloat.append({"x":self.converteBinToFloat(i["x"]), "y":self.converteBinToFloat(i["y"]), "fitness":i["fitness"]}) #Adiciona o individuo na lista de individuos em float

        plotIndList = self.getListOfDict(populacaoFloat) #Transforma a lista de dicionarios em uma lista de listas (Para manipular na hora de plotar o grafico)

        X, Y = np.meshgrid(np.array(plotIndList[0]), np.array(plotIndList[1])) #Cria a malha de pontos para o grafico
        calcFitness_vect = np.vectorize(self.calcFitness) 
        Z = calcFitness_vect(X, Y) #Calcula o fitness para cada ponto da malha

        plt.imshow(Z, cmap='hot', interpolation='nearest') #Plota um grafico de calor
        plt.colorbar(label='Fitness') #A barra de cores é a partir do fitness
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Fitness da população na geração {self.geracao}")
        plt.show()

    
    def plotarGraficoFitness(self, fitness_medio_por_geracao): #Plota o grafico do fitness médio por geração
        plt.plot(range(1, len(fitness_medio_por_geracao) + 1), fitness_medio_por_geracao)
        plt.title('Valor Médio do Fitness por Geração')
        plt.xlabel('Geração')
        plt.ylabel('Fitness Médio')
        plt.show()

        
    def printIndividuos(self, individuos:list):
        df = pd.DataFrame(individuos)
        print(df)
