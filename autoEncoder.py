import sys
from scipy import optimize
import numpy as np
from multicapa import *
from FontNew import *
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from red_entrenada import *

#def f(x):
#   return x**2
#minimum = optimize.fmin_powell(f, -1)


def stop():
    int(input(":>"))


def plot_graph(color):
    plt.plot(epocas, MSE, c=color, lw='1')


def main():

    a = [0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,0,1,1,0,1]
    b = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,0,0]
    c = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,1,0]
    d = [0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1]
    e = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1]
    f = [0,0,1,1,0,0,1,0,0,1,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0]
    g = [0,1,1,1,0,1,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,0]
    h = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,1,1,0,0,0,1,1,0,0,0,1]
    i = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0]
    j = [0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,1,1,0,0]
    k = [1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,0]
    l = [0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]

    #a_1b = [1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,0,1,1,0,1] #1 bit change
    #a_2b = [0,0,1,0,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,0,1,0,0,1] #2 bit change
    #a_3b = [0,1,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,1,0,0,1,1,0,1,1,0,1] #3 bit change 
    #a_4b = [0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,0,0,1,1,0,0] #4 bit change
    #a_5b = [1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,0,1,1,1,1] #5 bit change

    autoencoder = crear_red(35,[15,2,15],35)
    COTA_ERROR         = 0.001
    factor_aprendizaje = 0.005
    momentum           = 0.8
    epochs             = 40000

    stop()

    print("\n\n Entrenar \n\n")
    espacio_saliente, MSE, epocas = entrenar_red(autoencoder, Font, factor_aprendizaje, momentum, epochs, COTA_ERROR)
    print("\n\n Post entrenar \n\n")

    plot_graph('red')
    plt.ylabel('Error')
    plt.xlabel('Epocas')
    plt.show()
    plt.clf()


    #print(autoencoder)
    #stop()
    #autoencoder = red

    #prediction_1b, latente = predict(autoencoder, a_1b)
    #prediction_2b, latente = predict(autoencoder, a_2b)
    #prediction_3b, latente = predict(autoencoder, a_3b)
    #prediction_4b, latente = predict(autoencoder, a_4b)
    #prediction_5b, latente = predict(autoencoder, a_5b)

    #prediction_round_1b = np.rint(prediction_1b)
    #prediction_round_1b = prediction_round_1b.astype(int) #convert from float to int
    #prediction_round_2b = np.rint(prediction_2b)
    #prediction_round_2b = prediction_round_2b.astype(int) #convert from float to int
    #prediction_round_3b = np.rint(prediction_3b)
    #prediction_round_3b = prediction_round_3b.astype(int) #convert from float to int
    #prediction_round_4b = np.rint(prediction_4b)
    #prediction_round_4b = prediction_round_4b.astype(int) #convert from float to int
    #prediction_round_5b = np.rint(prediction_5b)
    #prediction_round_5b = prediction_round_5b.astype(int) #convert from float to int

    #print('a         : {0}'.format(a))
    #print('a_1b  : {0}'.format(list(prediction_round_1b)))
    #print("||a - a_1b|| = ", np.linalg.norm(np.array(a) - np.array(prediction_1b)))
    #print('\n')

    #print('a     : {0}'.format(a))
    #print('a_2b  : {0}'.format(list(prediction_round_2b)))
    #print("||a - a_2b|| = ", np.linalg.norm(np.array(a) - np.array(prediction_2b)))
    #print('\n')

    #print('a     : {0}'.format(a))
    #print('a_3b  : {0}'.format(list(prediction_round_3b)))
    #print("||a - a_3b|| = ", np.linalg.norm(np.array(a) - np.array(prediction_3b)))
    #print('\n')

    #print('a     : {0}'.format(a))
    #print('a_4b  : {0}'.format(list(prediction_round_4b)))
    #print("||a - a_4b|| = ", np.linalg.norm(np.array(a) - np.array(prediction_4b)))
    #print('\n')

    #print('a     : {0}'.format(a))
    #print('a_5b  : {0}'.format(list(prediction_round_5b)))
    #print("||a - a_5b|| = ", np.linalg.norm(np.array(a) - np.array(prediction_5b)))
    #print('\n')



    #x = []
    #y = []
    #i=0
    #for i in range(len(espacio_saliente)):
    #    x.append(espacio_saliente[i][0])
    #    y.append(espacio_saliente[i][1])
    #plt.scatter(x, y)

    #plt.show()


if __name__ == '__main__':
    main()


# 1) Entrenamiento:
#     Entrada -> Encoder -> EspacioLatente -> Decoder -> salida
#     1.bis) Error=Salida-Entrada -> backtracking() -> Actualiza pesos()
# Repetimos 1) hasta entrenar todo el conjunto.

