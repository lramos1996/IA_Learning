from random import choice
from math import sqrt,tanh
import numpy as np

multiply = lambda a,b: np.array(a).dot(np.array(b).transpose()).tolist()

class NeuralNetwork:
    def __init__(self, layers, fcActList,derFcActList):
        for i in range(len(layers)-1): layers[i]+=1 # generar bias
        self.layers = np.array(layers)
        self.fcActList = fcActList
        self.derFcActList = derFcActList
        self.TotalECM,self.long = 0,0
        #[[[][][]][[][][]][[][][][]]]
        #lista = [float(a)/100 for a in range(-100,101,1)]
        self.delt, self.funcion, self.zeta, self.derfuncion = ([np.ones(capa)  for capa in self.layers] for _ in range(4))
        print(f"zeta : {self.zeta},\n funcion : {self.funcion},\n derfuncion : {self.derfuncion}")
        #inicializar Wij
        self.weight = [np.random.rand(self.layers[i]*(self.layers[i+1]-1)) if i < self.layers.size-2 else np.random.rand(self.layers[i]*self.layers[i+1]) for i in range(self.layers.size-1)]
        print("Wij: ",self.weight)
        self.weightMin,self.ecmMin,  self.weightIni  = [], 100, []
        #np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
    def getWeight(self):
        return self.weight
    def train(self,x_train,y_train,epoch,learnRate):
        #print(y_train)
        tamanio,tamY,self.long = len(x_train), len(y_train[0]),len(x_train)
        for iter in range(epoch):
            ecm = 0
            if iter == 0 :
                print("Hola")
                self.weightIni = self.weight
            for m in range(tamanio):
                #evaluacion
                #print(y_train[j])
                self.funcion[0] = np.array(x_train[m])
                for i in range(1, self.layers.size - 1):
                    #print(self.zeta[i].size-1,self.zeta[i-1].size, self.funcion[i-1].reshape(self.funcion[i-1].size,1).shape)
                    self.zeta[i][:self.zeta[i].size - 1] = self.weight[i - 1].reshape(self.zeta[i].size - 1,
                                                                                      self.zeta[i - 1].size).dot(
                        self.funcion[i - 1].reshape(self.funcion[i - 1].size, 1)).transpose()
                    for j in range(self.funcion[i].size - 1): self.funcion[i][j] = self.fcActList[i](self.zeta[i][j])
                    for j in range(self.derfuncion[i].size - 1): self.derfuncion[i][j] = self.derFcActList[i](self.zeta[i][j])
                self.zeta[self.layers.size - 1] = self.weight[self.layers.size - 2].reshape(
                    self.zeta[self.layers.size - 1].size, self.zeta[self.layers.size - 2].size).dot(
                    self.funcion[self.layers.size - 2].reshape(self.funcion[self.layers.size - 2].size, 1)).transpose()
                aplicar = np.vectorize(self.fcActList[self.layers.size - 1])
                derivaraplicar = np.vectorize(self.derFcActList[self.layers.size - 1])
                self.funcion[self.layers.size - 1] = aplicar(self.zeta[self.layers.size - 1])
                self.derfuncion[self.layers.size - 1] = derivaraplicar(self.zeta[self.layers.size - 1])
                #for j in range(self.funcion[self.layers.size - 1].size): self.funcion[self.layers.size - 1][j] = self.fcActList[self.layers.size - 1](self.zeta[self.layers.size - 1][j])
                #for j in range(self.derfuncion[self.layers.size - 1].size): self.derfuncion[self.layers.size - 1][j] = \
                #self.derFcActList[self.layers.size - 1](self.zeta[self.layers.size - 1][j])
                """self.funcion[0]=np.array(x_train[m])
                for i in range(1,self.layers.size):
                    k=0
                    if i < self.layers.size-1 : #evaluacion de las capas ocultas
                        for j in range(self.funcion[i].size-1):
                            print(self.funcion[i-1].shape,self.weight[i-1][k:k+self.funcion[i-1].size].transpose().shape)
                            self.zeta[i][j] = self.funcion[i-1].dot(self.weight[i-1][k:k+self.funcion[i-1].size].transpose())
                            self.funcion[i][j] = self.fcActList[i](self.zeta[i][j])
                            self.derfuncion[i][j] = self.derFcActList[i](self.zeta[i][j])
                            k+=self.funcion[i-1].size
                    elif i == self.layers.size-1 : #evaluacion de la capa de salida
                        for j in range(self.funcion[i].size):
                            self.zeta[i][j] = self.funcion[i-1].dot(self.weight[i-1][k:k+self.funcion[i-1].size].transpose())
                            self.funcion[i][j] = self.fcActList[i](self.zeta[i][j])
                            self.derfuncion[i][j] = self.derFcActList[i](self.zeta[i][j])
                            k+=self.funcion[i-1].size"""
                #backpropagation
                error = y_train[m] - self.funcion[self.layers.size-1]
                self.delt[self.layers.size-1] = error*self.derfuncion[self.layers.size-1]
                #print("prueba: ",self.delt[self.layers.size-1].size, self.delt[self.layers.size-2].size,self.delt[self.layers.size-1].shape)
                self.delt[self.layers.size-2] = self.delt[self.layers.size-1].dot(self.weight[self.layers.size-2].reshape(self.delt[self.layers.size-1].size, self.delt[self.layers.size-2].size))
                for i in range(self.layers.size - 3, -1, -1):
                    #print(self.delt[i+1][:self.delt[i+1].size-1].shape,self.weight[i].reshape(self.delt[i+1].size-1, self.delt[i].size).shape,"shape: ",self.delt[i+1][0][:self.delt[i+1].size-1].reshape(1,self.delt[i+1][0][:self.delt[i+1].size-1].size).shape)
                    self.delt[i] = self.delt[i+1][0][:self.delt[i+1].size-1].reshape(1,self.delt[i+1][0][:self.delt[i+1].size-1].size).dot(self.weight[i].reshape(self.delt[i+1].size-1, self.delt[i].size))
                self.weight[len(self.weight) - 1] += self.funcion[len(self.weight) - 1].reshape(self.funcion[len(self.weight) - 1].size,1).dot(self.delt[len(self.weight)].reshape(1,self.delt[len(self.weight)].size)).transpose().ravel()*learnRate
                for i in range(len(self.weight) - 2, -1, -1): self.weight[i] += self.funcion[i].reshape(self.funcion[i].size,1).dot(self.delt[i+1][0][:self.delt[i+1].size-1].reshape(1,self.delt[i+1][0][:self.delt[i+1].size-1].size)).transpose().ravel()*learnRate
                #Calculo de error por iteracion
                ecm += (np.sum(error))**2
            #calculo error de cada epoch
            self.TotalECM = sqrt(ecm / (2 * tamanio))
            if self.TotalECM < self.ecmMin :
                self.weightMin = self.weight
                self.ecmMin = self.TotalECM
            ecm = 0
            print("Iter N° :",iter,"ErrorCM : ", self.TotalECM)
            #for m in range(self.long): print("Y_Real :",y_train[m],"       Y_Predecido :",self.probar(x_train[m]))
            #print(f"zeta : {self.zeta},\nfuncion : {self.funcion},\nWij : {self.weight}\nDelta : {self.delt} ")

    def result(self,x_train,y_train):
        print("ENTRENADO!\n")
        for m in range(self.long): print("Y_Real :",y_train[m],"       Y_Predecido :",self.probar(x_train[m]))
        #print(self.layers,type(self.layers))
        #print([a.size for a in self.weight])
        print("Valores finales!\n")
        print(f"zeta : {self.zeta},\nfuncion : {self.funcion},\nderfuncion : {self.derfuncion}\nWij : {self.weight}\nDelta : {self.delt}\nEcmMin : {self.ecmMin}\nWeightMin : {self.weightMin}\nWeightInicial : {self.weightIni}")
    def probar(self,entrada):
        self.funcion[0] = np.array(entrada)
        for i in range(1, self.layers.size - 1):
            # print(self.zeta[i].size-1,self.zeta[i-1].size, self.funcion[i-1].reshape(self.funcion[i-1].size,1).shape)
            self.zeta[i][:self.zeta[i].size - 1] = self.weight[i - 1].reshape(self.zeta[i].size - 1,
                                                                              self.zeta[i - 1].size).dot(
                self.funcion[i - 1].reshape(self.funcion[i - 1].size, 1)).transpose()
            for j in range(self.funcion[i].size - 1): self.funcion[i][j] = self.fcActList[i](self.zeta[i][j])
            for j in range(self.derfuncion[i].size - 1): self.derfuncion[i][j] = self.derFcActList[i](self.zeta[i][j])
        self.zeta[self.layers.size - 1] = self.weight[self.layers.size - 2].reshape(
            self.zeta[self.layers.size - 1].size, self.zeta[self.layers.size - 2].size).dot(
            self.funcion[self.layers.size - 2].reshape(self.funcion[self.layers.size - 2].size, 1)).transpose()
        aplicar = np.vectorize(self.fcActList[self.layers.size - 1])
        derivaraplicar = np.vectorize(self.derFcActList[self.layers.size - 1])
        self.funcion[self.layers.size - 1] = aplicar(self.zeta[self.layers.size - 1])
        self.derfuncion[self.layers.size - 1] = derivaraplicar(self.zeta[self.layers.size - 1])
        #print(f"zeta : {self.zeta},\nfuncion : {self.funcion}")
        return self.funcion[self.layers.size - 1][0]
        #print("Salida : ", self.funcion[self.layers.size - 1])
        """for i in range(1,self.layers.size-1):
            #print(self.zeta[i].size-1,self.zeta[i-1].size, self.funcion[i-1].reshape(self.funcion[i-1].size,1).shape)
            self.zeta[i][:self.zeta[i].size-1] = self.weight[i-1].reshape(self.zeta[i].size-1,self.zeta[i-1].size).dot(self.funcion[i-1].reshape(self.funcion[i-1].size,1)).transpose()
            for j in range(self.funcion[i].size-1): self.funcion[i][j] = self.fcActList[i](self.zeta[i][j])
        self.zeta[self.layers.size-1] = self.weight[self.layers.size-2].reshape(self.zeta[self.layers.size-1].size,self.zeta[self.layers.size-2].size).dot(
            self.funcion[self.layers.size-2].reshape(self.funcion[self.layers.size-2].size,1)).transpose()
        for j in range(self.funcion[self.layers.size-1].size): self.funcion[self.layers.size-1][j] = self.fcActList[self.layers.size-1](self.zeta[self.layers.size-1][j])
        print("Salida : ",self.funcion[self.layers.size-1])"""
def regresion(x): return x
def devregresion(x): return 1
def f_activacion(x): return 1 if x > 1 else 0 if x > 0 and x < 1 else -1
def f_xor(x) : return 1 if x>0 else 0
def devf_xor(x) : return 0 if x>1 else 1 if x<1 and x>-1 else 0
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-5*x))
def sigmoid_derivada(x):
    return 5 * sigmoid(x) * (1.0 - sigmoid(x))
def tanhiper(x):
    return tanh(x)
def dertanhiper(x):
    return 1 -tanhiper(x)**2
def getX(dataset,tamvarY):
    n= len(dataset[0])
    salida = []
    for a in dataset:
        aux=a[0:n-tamvarY]
        #for i in range(tamvarY): aux.append(1)
        aux.append(1)
        salida.append(aux)
    return salida
def getY(dataset,tamvarY):
    return [np.array(dat[len(dataset[0]) - tamvarY: len(dataset[0])]) for dat in dataset]
def datasetCompuertasAnd():
    #return [[1,1,1],[1,0,0],[0,1,0],[0,0,0]]
    #return [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
    #return [[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]]
    return [[0, 0, 0,1], [0, 1, 0,1], [0, -1, 0,1], [0.5, 1, -1,1],[0.5,-1, 1,1],[1,1, 0,-1],[1,-1, 0,-1]]
if __name__ == "__main__":
    #entrenar carrito
    nn = NeuralNetwork([2,3,3,2],[tanhiper,tanhiper,tanhiper,tanhiper],[dertanhiper,dertanhiper,dertanhiper,dertanhiper])
    #entrenar xor
    #nn = NeuralNetwork([2,2, 1],[f_xor,f_xor,f_xor],[devregresion,devregresion,devregresion])
    dataset = datasetCompuertasAnd()
    tamvary = 2
    x_train = getX(dataset,tamvary)
    y_train = getY(dataset,tamvary)
    print(f"x_train : {x_train}\n y_train : {y_train}")
    #train carrito
    nn.train(x_train,y_train,100000,0.003)
    #train xor
    #nn.train(x_train, y_train, 800, 0.0000001)
    nn.result(x_train,y_train)
    opcion = int(input("0 para cancelar otro num para continuar\nIngrese Opcion : "))
    while opcion != 0 :
        #entrada = [int(input()) for _ in range(len(y_train)-tamvary)]
        entrada = list(map(lambda k: int(k),input(f"Ingrese {len(x_train[0])-tamvary} numeros separado por espacios : ").split()))
        entrada.append(1)
        print("Salida :", nn.probar(entrada))
        opcion = int(input("0 para cancelar otro num para continuar\nIngrese Opcion : "))

    """
    Wij : [array([-29.93751147,  -5.2885016 ,   9.47790981,   8.26022505,
        -2.03035196,  -5.8530215 ,   8.13435776,  -1.97081794,
        -6.45336103]), array([-9.20267831,  3.75966987,  4.46197766, -1.33373058,  0.68904197,
       -5.52373653, -1.45900383,  1.19639763,  0.27658297,  0.97798563,
       -0.49933449,  0.08516305]), array([-1.13195491, -0.26392569,  4.3842401 , -0.35844292, -3.15450874,
        1.71506007,  2.49964613,  1.21724232])]
    """
    """
    wij final de xor:
    Wij : [array([-0.15667499,  0.26437559, -0.1082188 ,  0.23588067, -0.21987573,
       -0.01654719]), array([ 0.00929466,  0.00722091, -0.00229903])]
    """