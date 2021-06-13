from multicapa import *
from font3 import *
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statistics import mean
import statistics
import numpy as np
from functools import reduce
import sys



def main():
    a = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]
    # encoder = crear_red(35, [5], 2)
    decoder = crear_red(2, [5], 35)
    autoencoder = crear_red(35, [5, 2, 5], 35)

    print("\n\n Entrenar \n\n")
    espacio_saliente = entrenar_red(autoencoder, Font3, 0.01, 5000, 2, 0.002)
    print("\n\n Post entrenar \n\n")
    print(espacio_saliente)
    print(len(espacio_saliente))
    # prediction = predict(autoencoder, a)
    x = []
    y = []
    i=0
    for i in range(len(espacio_saliente)):
        x.append(espacio_saliente[i][0])
        y.append(espacio_saliente[i][1])
    plt.scatter(x, y)

    plt.show()


if __name__ == '__main__':
    main()


# 1) Entrenamiento:
#     Entrada -> Encoder -> EspacioLatente -> Decoder -> salida
#     1.bis) Error=Salida-Entrada -> backtracking() -> Actualiza pesos()
# Repetimos 1) hasta entrenar todo el conjunto.

