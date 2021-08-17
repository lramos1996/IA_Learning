from random import choice
from math import sqrt
import time
def crearDataSetRandom(num):
    lista = [i for i in range(num)]
    # a = choice([i for i in range(num)])
    cont, data = 0, []
    while cont < num:
        a = choice(lista)
        #if (a, 32 + a * 9 / 5) not in data:
        if (a, 24.56 + a * 0.763) not in data:
            #data.append((a, 32 + a * 9 / 5))
            data.append((a, 24.56 + a * 0.763))
            lista.remove(a)
            cont += 1
    return data


def crearDataSetOrd(num):
    return [(a, 32 + a * 9 / 5) for a in range(num)]


def imprimirData(data):
    print("len data:", len(data))
    print("Posicion ", " (C°, F°)")
    for i in range(len(data)):
        print("pos:", i, " ", data[i])


def recorrido(data, factTrain, a, b):
    for dat in data:
        y = dat[0] * a + b
        error = dat[1] - y
        delta1 = factTrain * error * dat[0]
        delta2 = factTrain * error
        a = a + delta1
        b = b + delta2
        # print ("a =",a,"b =",b, "delta1 =",delta1, "delta2 =",delta2, "y =",y, "Error =",error)
        return a, b


def Intrain(data, factTrain, num, nroiter):
    #entrenamiento = int(0.80 * len(data))
    #prueba = len(data) - entrenamiento
    #train = data[0:entrenamiento]
    #proof = data = data[entrenamiento:entrenamiento + prueba]
    lista, i , ecm , n= [i for i in range(num)], 0, 0, len(data)
    a, b = choice(lista), choice(lista)
    #print("ainicial =",a,"binicial =",b)
    while i < nroiter:
        print("Entrenamiento n°:", i)
        for dat in data:
            #print(a,b)
            y = dat[0] * a + b
            error = dat[1] - y
            if abs(error) < 0.0000000005: return a, b
            delta1 = factTrain * error * dat[0]
            delta2 = factTrain * error * 1
            #print(a, b,delta1,delta2)
            a = a + delta1
            b = b + delta2
            ecm += error**2
            #print("a =", a, "b =", b, "ydata =", y, "yreal =", dat[1], "Error =", error)
        #if sqrt(ecm/nroiter) < 4.1 : return a, b
        print("ECM =",sqrt(ecm/(2*n)))
        ecm = 0
        i += 1
    return a, b
    print("Finalp a =", a, "Final b =", b)

if __name__=="__main__":
    Data,n = crearDataSetRandom(1000),0
    a, b = 0, 0
    print("BIENVENIDO AL IA QUE CONVIERTE C° A F°")
    print("LR.soft SA")
    print("______________________________________")
    print("Presione 1 para entrenar :°")
    print("Presione 2 para probar :°")
    print("Presione 3 para salir :°")
    while True:
        try:
            n = int(input("Ingrese lo que quiere hacer: "))
            break
        except  ValueError:
            print ("Debe ser un valor NUMERICO!: ")
    print("______________________________________")
    while n != 3 :
        while n == 2 and a+b==0:
            print("Falta entrenar el modelo!")
            print("Presione 1 para entrenar :°")
            while True:
                try:
                    n = int(input("Ingrese lo que quiere hacer: "))
                    break
                except  ValueError:
                    print("Debe ser un valor NUMERICO!: ")
            print("______________________________________")
        if n == 1 :
            t = time.time()
            a,b=Intrain(Data, 0.000001, 50, 100000)
            z = time.time()
            print("Final a =", a, "Final b =", b)
            print("Tiempo entrenamiento:", z - t)
            print("Modelo entrenado!")
            print("______________________________________")
            print("Presione 1 para volver entrenar :°")
            print("Presione 2 para probar :°")
            print("Presione 3 para salir :°")
            while True:
                try:
                    n = int(input("Ingrese lo que quiere hacer: "))
                    break
                except  ValueError:
                    print("Debe ser un valor NUMERICO!: ")
            print("______________________________________")
        if n == 2 and a+b !=0:
            print("Presione 1 para lista:")
            print("Presione 2 para probar numero:")
            print("Presione 3 para salir :°")
            while True:
                try:
                    p = int(input("Ingrese lo que quiere hacer: "))
                    break
                except  ValueError:
                    print("Debe ser un valor NUMERICO!: ")
            print("______________________________________")
            if p == 1 :
                for i in Data:
                    print("F°_real= ", i[1], " F°_generado= ", a * i[0] + b)
                print("______________________________________")
            elif p == 2 :
                s = 0
                while s != 3 :
                    while True:
                        try:
                            m = float(input("Ingrese numero C° para probar: "))
                            break
                        except  ValueError:
                            print("Debe ser un valor NUMERICO!: ")
                    #print("F°_real= ", 32+m*9/5, " F°_generado= ", a * m + b)
                    print("F°_real= ", 24.56 + m * 0.763, " F°_generado= ", a * m + b)
                    while True:
                        try:
                            s = int(input("Ingrese 0 para repetir, 3 para salir: "))
                            break
                        except  ValueError:
                            print("Debe ser un valor NUMERICO!: ")
                    print("______________________________________")
            elif p == 3 :
                print("______________________________________")
                print("Presione 1 para volver entrenar :°")
                print("Presione 2 para probar :°")
                print("Presione 3 para salir :°")
                while True:
                    try:
                        n = int(input("Ingrese lo que quiere hacer: "))
                        break
                    except  ValueError:
                        print("Debe ser un valor NUMERICO!: ")
                print("______________________________________")
    print("GRACIAS POR USAR EL PROGRAMA!")
#a = 1.7976522524106076 Final b = 32.78179413027762
#a = 1.7891568409634173 Final b = 33.66872910055728

#a = 3.211555596701065 Final b = 29.373193912855626
#Final a = 1.799999999348894 Final b = 32.000000392465736
#a = 3.845408976808621   b=26.952909742671906
#Tiempo entrenamiento: 109.32503175735474