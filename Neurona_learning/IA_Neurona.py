from random import choice
from math import sqrt
import time
class neurona :
    def __init__(self):
        self.x = []
        self.w = []
    def __generarRamdom(self,num):
        lista = [a for a in range(num)]
        return choice(lista)
    def __multiplicar(self,a,b): return sum(map(lambda x:x[0]*x[1],zip(a,b)))
    def __evaluar(self, funcion,x): return funcion(x)
    def entrenar(self,data,funcion,derfuncion,rangeRandom, factTrain, nroiter):
        self.nIndice = len(data)
        self.dataSet = data
        i, ecm, n,tam, aux,z,y = 0, 0, self.nIndice,len(self.dataSet[0]),[],0,0
        self.w = [self.__generarRamdom(rangeRandom) for _ in range(tam)]
        self.__setX(self.dataSet, tam)
        y = [dat[tam - 1] for dat in self.dataSet]
        print("inicial",self.__getW())
        while i < nroiter:
            print("Entrenamiento n°:", i)
            j = 0
            for dat in self.dataSet:
                #y = dat[tam - 1]
                #aux=dat
                #print(dat)
                #self.__setX(aux,tam)
                #print(dat)
                z = self.__multiplicar(self.__getX(j),self.__getW())
                #print(self.__getX(),self.__getW())
                f = self.__evaluar(funcion,z)
                #y = dat[0] * a + b
                #error = dat[1] - y
                error = y[j]-f
                #print(f,y,error)
                #print(error)
                #if abs(error) < 0.0000000005: return self.__getW()
                delta = [ factTrain * error * self.__getX(j)[i]* self.__evaluar(derfuncion,z) for i in range(tam)]
                #delta1 = factTrain * error * dat[0]
                #delta2 = factTrain * error * 1
                #print(factTrain, error, self.__evaluar(derfuncion,z),delta)
                #print(self.__getW())
                #print(self.__getW(),delta,[x + y for x, y in zip(self.__getW(),delta)])
                self.__setW([x + y for x, y in zip(self.__getW(),delta)])
                #print(self.__getW())
                #a = a + delta1
                #b = b + delta2
                ecm += error ** 2
                j += 1
                # print("a =", a, "b =", b, "ydata =", y, "yreal =", dat[1], "Error =", error)
            # if sqrt(ecm/nroiter) < 4.1 : return a, b
            print("ECM =", sqrt(ecm / (2 * n)))
            if sqrt(ecm / (2 * n))< 0.0000000005: return self.__getW()
            ecm = 0
            i += 1
        print("ENTRENADO!")
        return self.__getW()
        #print("Finalp a =", a, "Final b =", b)
    def __setX(self,lista,n):
        for a in lista:
            aux=a[0:n-1]
            aux.append(1)
            self.x.append(aux)
        #print(self.x)
    def __getX(self,m): return self.x[m]
    def __setW(self,lista):
        self.w = lista
    def __getW(self): return self.w

def regresion(x): return x
def devregresion(x): return 1
def f_activacion(x): return 1 if x > 0 else 0
def crearDataSetRandom(num):
    lista = [i for i in range(num)]
    # a = choice([i for i in range(num)])
    cont, data = 0, []
    while cont < num:
        a = choice(lista)
        if (a, 32 + a * 9 / 5) not in data:
            data.append([a, 32 + a * 9 / 5])
            lista.remove(a)
            cont += 1
    return data
def datasetCompuertasAnd():
    return [[1,1,1],[1,0,0],[0,1,0],[0,0,0]]
def datasetCompuertasOr():
    return [[1,1,1],[1,0,1],[0,1,1],[0,0,0]]
def datasetRegre():
    return [[1,1,1],[1,0,1],[0,1,1],[0,0,0]]
if __name__ == "__main__":
    #data = crearDataSetRandom(1000)
    cerebro = neurona()
    #a=time.time()
    #listaW = cerebro.entrenar(data,regresion,devregresion,50,0.000001, 100000)
    #b=time.time()
    #print(listaW)
    #print("Tiempo entrenamiento:",b-a)
    #tiempo entrenamiento alg 1 : 689.368800163269 s

    #Prueba con compuertas logicas
    data = datasetCompuertasAnd()
    a = time.time()
    listaW = cerebro.entrenar(data,f_activacion,devregresion,2,0.0001, 30000)
    b = time.time()
    print(listaW)
    print("Tiempo entrenamiento:", b - a)
    #Prueba con regresion
    """data = datasetRegre()
    a = time.time()
    listaW = cerebro.entrenar(data,regresion,devregresion,10,0.0001, 100000)
    b = time.time()
    print(listaW)
    print("Tiempo entrenamiento:", b - a)"""