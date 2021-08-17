import numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork :
    def __init__(self):
        self.layer,self.actFuncion, self.derActFunc, self.weight, self.funcion, self.derFuncion, self.zeta, self.delta = ([] for _ in range(8))
        self.actAndDerActFuncList = {"ReLu" : "derRelu", "regresion": "derRegresion", "tanh": "derTanh", "sigmoid": "derSigmoid", "clasBin" : "derClasBin","clasTanh":"derClasTanh"}
        self.histError = []
    def inputLayer(self, num_neuron = 0):
        self.layer.append(num_neuron)
    def addLayer(self, num_neuron = 0, activation = 'ReLu'):
        self.layer.append(num_neuron)
        self.actFuncion.append(np.vectorize(eval(f"self.{activation}")))
        self.derActFunc.append(np.vectorize(eval(f"self.{self.getDerFuncAct(activation)}")))
        self.weight.append(np.random.rand(self.layer[len(self.layer)-1]*(self.layer[len(self.layer)-2]+1)))
        self.weight[len(self.weight)-1]=self.weight[len(self.weight)-1].reshape(self.layer[len(self.layer)-1],(self.layer[len(self.layer)-2]+1))
    #Proceso de aprendizaje
    def process(self, entrada):
        self.cleanNeuron()
        self.funcion.append(np.atleast_2d(entrada))
        for i in range(1,len(self.layer)) :
            self.zeta.append(np.atleast_2d(np.concatenate((self.funcion[i-1], [1.]), axis=None)).dot(self.weight[i-1].transpose()))
            self.funcion.append(self.actFuncion[i-1](self.zeta[i-1]))
            self.derFuncion.append(self.derActFunc[i-1](self.zeta[i-1]))
        return self.funcion[len(self.funcion)-1]
    def learn(self, trainX , trainY,learRate):
        #print(self.delta,len(self.delta)-1 )
        #Calculo de DELTAS
        y_predicted = self.process(trainX)
        error = np.atleast_2d(trainY) - y_predicted
        self.delta[len(self.delta)-1] = error.dot(np.diag(self.derFuncion[len(self.delta)-1][0]))
        for i in range(len(self.delta)-2,-1,-1): self.delta[i] = (self.delta[i+1].dot(self.weight[i+1][:,:self.weight[i+1].shape[1] - 1])).dot(np.diag(self.derFuncion[i][0]))
        #Calculo de nuevos Pesos
        for i in range(len(self.delta) - 1, -1, -1): self.weight[i]+=learRate*self.delta[i].T.dot(np.atleast_2d(np.concatenate((self.funcion[i], [1.]), axis=None)))
        #ecm = np.sum(np.vectorize(lambda x : (x**2)/2)(error))
        return np.sum(np.vectorize(lambda x : (x**2)/2)(error))
    def train(self, trainX = [], trainY = [], learRate = 0.01, epochs = 100):
        tamDataSet = len(trainX)
        for i in range(epochs):
            TotalEcm = 0
            for j in range(tamDataSet): TotalEcm += self.learn(trainX[j],trainY[j],learRate)
            self.histError.append(TotalEcm/tamDataSet)
            print("Iter N° :", i, "ErrorCM : ", TotalEcm/tamDataSet)
        print("MODELO ENTRENADO!\n")
    #Limpiar neurona
    def cleanNeuron(self):
        self.funcion, self.derFuncion, self.zeta = ([] for _ in range(3))
        self.delta = [[] for _ in range(len(self.layer)-1)]
    #Graficar entrenamiento
    def graphTrain(self,typeGraph = 'plot',escala = [0, 1],dat = 'self.histError'):
        datos = eval(dat)
        fig, ax = plt.subplots()
        if escala != [] : plt.ylim(escala)
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        eval(f"ax.{typeGraph}({range(len(datos))}, {datos}, color='b')")
        plt.tight_layout()
        plt.show()
    #Funciones de activacion
    def ReLu(self, x): return x if x > 0 else 0
    def regresion(self, x): return x
    def sigmoid(self, x): return 1.0 / (1.0 + np.exp(-5 * x))
    def tanh(self, x): return np.tanh(x)
    def clasBin(self, x): return 1 if x>0 else 0
    def clasTanh(self, x): return 1 if x>0.5 else 0 if x<0.5 and x>-0.5 else -1
    #Funciones derivadas de activacion
    def getDerFuncAct(self, funcion): return self.actAndDerActFuncList[funcion]
    def setDerFuncAct(self, funcion, derfuncion): self.actAndDerActFuncList[funcion] = derfuncion
    def derRegresion(self,x): return 1
    def derRelu(self,x): return 1 if x > 0 else 0
    def derSigmoid(self, x): return 5 * self.sigmoid(x) * (1.0 - self.sigmoid(x))
    def derTanh(self, x): return 1 - np.tanh(x) ** 2
    def derClasBin(self,x): return 1
    def derClasTanh(self,x): return 1
    #Obtener lista de funciones de activacion
    def getActFuncList(self):
        print("\u0332".join("Funcion | Derivada Funcion "))
        for key in self.actAndDerActFuncList: print(key, "|", self.actAndDerActFuncList[key])
    #Crear dataset x
    def getTrainX(self, dataset, tamvarY = 1):
        n,salida = len(dataset[0]),[]
        for a in dataset:
            aux = a[0:n - tamvarY]
            salida.append(aux)
        return salida
    # Crear dataset Y
    def getTrainY(self, dataset, tamvarY = 1): return [dat[len(dataset[0]) - tamvarY: len(dataset[0])] for dat in dataset]
    # Mostrar Resultados
    def showResult(self, x , y ):
        for i in range(len(x)): print("Yreal =", y[i], " ", "Ypredecido =", self.process(x[i])[0])
        print("Matrices de pesos:")
        for matriz in self.weight : print(matriz, end = '\n\n')

def datasetCompuertasAnd():
    #return [[1,1,1],[1,0,0],[0,1,0],[0,0,0]]
    #return [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    return [[0, 0, 0, 1], [0, 1, 0, 1], [0, -1, 0, 1], [0.5, 1, -1, 1], [0.5, -1, 1, 1], [1, 1, 0, -1], [1, -1, 0, -1]]

if __name__ == "__main__":

    brain = NeuralNetwork()
    brain.getActFuncList()
    brain.inputLayer(2)
    brain.addLayer(num_neuron=3,activation='ReLu')
    brain.addLayer(num_neuron=2,activation='tanh')
    brain.addLayer(num_neuron=2, activation='tanh')
    #print(brain.layer)
    #print(brain.actFuncion[0](np.array([3,4])))
    #print(brain.derActFunc[0](np.array([3, 4])))
    #print(brain.weight)
    ##print(brain.derFuncion)
    #print(brain.learn([2,4],[3,4],0.1))
    data = datasetCompuertasAnd()
    tamy = 2
    trainx = brain.getTrainX(data, tamy)
    trainy = brain.getTrainY(data, tamy)
    brain.train(trainx,trainy,learRate=0.03,epochs=7500)
    brain.showResult(trainx,trainy)
    p =  sum(brain.histError)/len(brain.histError)
    brain.graphTrain(typeGraph='bar',escala=[0,p])