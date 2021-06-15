from multicapa import *
from font3 import *
from FontNew import *
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statistics import mean
import statistics
import numpy as np
from functools import reduce
import sys

def stop():
    int(input(":>"))

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

    autoencoder = crear_red(35,[15,2,15],35)

    COTA_ERROR         = 0.005
    factor_aprendizaje = 0.02
    epochs             = 40000

    print("\n\n Entrenar \n\n")
    espacio_saliente = entrenar_red(autoencoder, Font, factor_aprendizaje, epochs, COTA_ERROR)
    print("\n\n Post entrenar \n\n")


    #Predizer a,b,c,d
    prediction_a, latente = predict(autoencoder, a)
    prediction_b, latente = predict(autoencoder, b)
    prediction_c, latente = predict(autoencoder, c)
    prediction_d, latente = predict(autoencoder, d)

    prediction_round_a = np.rint(prediction_a)
    prediction_round_a = prediction_round_a.astype(int) #convert from float to int

    prediction_round_b = np.rint(prediction_b)
    prediction_round_b = prediction_round_b.astype(int) #convert from float to int

    prediction_round_c = np.rint(prediction_c)
    prediction_round_c = prediction_round_c.astype(int) #convert from float to int

    prediction_round_d = np.rint(prediction_d)
    prediction_round_d = prediction_round_d.astype(int) #convert from float to int

    print('a     : {0}'.format(a))
    print('a_hat : {0}'.format(list(prediction_round_a)))
    print("||a - a_hat|| = ", np.linalg.norm(np.array(a) - np.array(prediction_round_a)))
    print('\n')

    print('b     : {0}'.format(b))
    print('b_hat : {0}'.format(list(prediction_round_b)))
    print("||b - b_hat|| = ", np.linalg.norm(np.array(b) - np.array(prediction_round_b)))
    print('\n')

    print('c     : {0}'.format(c))
    print('c_hat : {0}'.format(list(prediction_round_c)))
    print("||c - c_hat|| = ", np.linalg.norm(np.array(c) - np.array(prediction_round_c)))
    print('\n')

    print('d     : {0}'.format(d))
    print('d_hat : {0}'.format(list(prediction_round_d)))
    print("||d - d_hat|| = ", np.linalg.norm(np.array(d) - np.array(prediction_round_d)))



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

