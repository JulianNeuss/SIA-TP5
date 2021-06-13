from random import seed
from random import random
from math import exp
import numpy as np

def crear_red(num_entradas, camada_escondida, num_salidas):

	red = list()
	capa_escondida = [{'pesos': [random() for i in range(num_entradas + 1)]} for i in range(camada_escondida[0])]
	red.append(capa_escondida)

	for h in range(len(camada_escondida) - 1):
		capa_escondida = [{'pesos': [random() for i in range(camada_escondida[h] + 1)]} for i in range(camada_escondida[h + 1])]
		red.append(capa_escondida)

	capa_salida = [{'pesos': [random() for i in range(camada_escondida[-1] + 1)]} for i in range(num_salidas)]
	red.append(capa_salida)
	
	return red

def activacion(pesos, inputs):
	bias = pesos[-1] #El bias es el ultimo elemento
	activation = bias 
	for i in range(len(pesos)-1): #for i in [xi,yi]
		activation += pesos[i] * inputs[i] #x1*w1 + x2*w2 + ... + xn*wn + bias
	return activation

def tan_hiperbolica(activation, beta=0.5):
	#tangente hiperbolica
	return np.tanh(beta*activation)

def tanh_dx_dt(output, beta=0.5):
	#Derivada de la tangente hiperpolica
	#dg(x)/dx = beta * (1 - f(x)²)
	dg_dx = beta*(1 - output**2)
	return dg_dx

def propagacion(red, entrada):
	espacio_latente = []
	inputs = entrada # (xi, yi, salida_i)
	for capa in red: #Nivel de la red.
		n_entradas = []
		for neurona in capa: #
			#Unidad de salida alcanza un estado de exitación h_{i}^{mu}
			activation_ = activacion(neurona['pesos'], inputs) #Pesos de V1 (w11, w21, 1) con input (xi,yi,Salida_i)
			#La salida para el neurônio V1 = sigmoide(w11*x1 + w21*x2 + 1)
			neurona['salida'] = tan_hiperbolica(activation_) #Diap : 10/49
			if len(capa) == 2:
				espacio_latente.append(tan_hiperbolica(activation_))
			n_entradas.append(neurona['salida'])
		inputs = n_entradas

	return inputs, espacio_latente

def backtracking(red, expected, beta=0.5): 
	for i in reversed(range(len(red))): #Empezamos en la camada de salida
		capa = red[i] #O1, O2

		for j in range(len(capa)): 
			Neurona_J = capa[j] #O1 = {'pesos':[]}, O2 = {'pesos':[]}
			error = 0.0

			if i != len(red)-1: 
				for Neurona_i in red[i + 1]:
					error += (Neurona_i['pesos'][j] * Neurona_i['delta'])
			else: 
				error = expected[j] - Neurona_J['salida']

			Neurona_J['error'] = error
			Neurona_J['delta'] = error * tanh_dx_dt(Neurona_J['salida']) #diap 14/49 : ultima ecuación
			

def actualizar_pesos(red, row, factor_aprendizaje):
	for i in range(len(red)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neurona['salida'] for neurona in red[i - 1]]
		for neurona in red[i]:
			for j in range(len(inputs)):
				neurona['pesos'][j] += factor_aprendizaje * neurona['delta'] * inputs[j]
			neurona['pesos'][-1] += factor_aprendizaje * neurona['delta']


def entrenar_red(red, set_entrenamiento, factor_aprendizaje, epochs, num_salidas, COTA_ERROR):
	latentes = []
	espacio_latente = []
	count_epochs = 0
	Error_w = 1

	while (Error_w > COTA_ERROR):
		Error_w = 0 #Funcion de costo E(W)
		espacio_latente = []
		for row in set_entrenamiento: # (xi, yi, salida_i)
			salida, el = propagacion(red, row) #Entrada -> Salida

			espacio_latente.append(el)
			salida_deseada = row

			#Error entre la salida deseada e la salida calculada (Diap : 11/49)
			Error_w += 0.5*sum([(salida_deseada[i]-salida[i])**2 for i in range(len(salida_deseada))])
			backtracking(red, salida_deseada)
			actualizar_pesos(red, row, factor_aprendizaje)

		print('[*]epocas={0}, aprendizaje={1}, error={2}'.format(count_epochs, factor_aprendizaje, Error_w))

		count_epochs += 1
		if count_epochs == epochs:
			break

	return espacio_latente


def predict(red, entrada):
	outputs = propagacion(red, entrada)
	return float(outputs[0])



def main():

	num_entradas  = len(dataset_TRAIN2[0]) - 1 #35
	num_salidas   = 1
	capas_escondidas   = [4,3]
	COTA_ERROR         = 0.002 
	factor_aprendizaje = 0.01
	epochs             = 20000

	red = crear_red(num_entradas, capas_escondidas, num_salidas)
	entrenar_red(red, dataset_TRAIN2, factor_aprendizaje, epochs, num_salidas, COTA_ERROR)

	#Classificacion
	for data in dataset_TEST2:

		prediction_original = predict(red, data)
		expected = data[-1]

		#Si el error es < 10% (clasifico bien)
		error = abs(prediction_original - expected)
		prediction = prediction_original
		if error < 0.1:
			prediction = round(prediction_original)


		print("Esperado={0}, Calculado={1},Original={2}, Error={3}".format(data[-1], prediction, prediction_original, error))

if __name__ == '__main__':
    main()