#from random import seed
#from random import random
import random
from math import exp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def stop():
	int(input(':>'))

def crear_red(num_entradas, camada_escondida, num_salidas):

	def r():
		return random.uniform(-0.50, +0.50)

	red = list()
	capa_escondida = [{'pesos': [r() for i in range(num_entradas + 1)]} for i in range(camada_escondida[0])]
	red.append(capa_escondida)

	for h in range(len(camada_escondida) - 1):
		capa_escondida = [{'pesos': [r() for i in range(camada_escondida[h] + 1)]} for i in range(camada_escondida[h + 1])]
		red.append(capa_escondida)

	capa_salida = [{'pesos': [r() for i in range(camada_escondida[-1] + 1)]} for i in range(num_salidas)]
	red.append(capa_salida)
	
	return red

def activacion(pesos, inputs):

	bias = pesos[-1] #El bias es el ultimo elemento
	activation = bias*1 
	for i in range(len(pesos)-1): #for i in [xi,yi]
		activation += pesos[i] * inputs[i] #x1*w1 + x2*w2 + ... + xn*wn + bias
	return activation

def sigmoid(activation):
    return (1/(1+np.exp(-activation)))

def sigmoid_dx_dt(output):
	dg_dx = output*(1.0-output)
	return dg_dx

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
	inputs = entrada 

	for capa in red: #Nivel de la red.

		n_entradas = []

		for neurona in capa: #

			#Unidad de salida alcanza un estado de exitación h_{i}^{mu}
			activation_ = activacion(neurona['pesos'], inputs) #Pesos de V1 (w11, w21, 1) con input (xi,yi,Salida_i)
			#La salida para el neurônio V1 = sigmoide(w11*x1 + w21*x2 + 1)
			neurona['salida'] = tan_hiperbolica(activation_)

			if len(capa) == 2:
				espacio_latente.append(tan_hiperbolica(activation_))
			
			n_entradas.append(neurona['salida'])

		inputs = n_entradas

	return inputs, espacio_latente


def backtracking(red, expected):

	for i in reversed(range(len(red))): #Empezamos por la ultima camada

		capa = red[i]
		errors = list()

		if i != len(red)-1:
			for j in range(len(capa)):
				error = 0.0
				for neurona in red[i + 1]:
					error += (neurona['pesos'][j] * neurona['delta'])
				errors.append(error)
		else:

			for j in range(len(capa)):
				neurona = capa[j]
				errors.append(expected[j] - neurona['salida'])
				
		for j in range(len(capa)):
			neurona = capa[j]
			neurona['delta'] = errors[j] * tanh_dx_dt(neurona['salida'])

def actualizar_pesos(red, row, factor_aprendizaje, momentum, delta):

	for i in range(len(red)):

		if i != 0: #if not first layer
			t = [neurona['salida'] for neurona in red[i - 1]]
		else: #if first layer
			t = row #init t as the training example attributes

		#para cada neurona in layer, zip with a length of network variable ot access deltas
		for neurona, d in zip(red[i], range(0, len(red[i]))):

			for f in range(len(t)):
				neurona['pesos'][f] += factor_aprendizaje * float(t[f]) * neurona['delta']

				if delta is not None:
					neurona['pesos'][f] += momentum * delta[d]

				#also update neural bias
				neurona['pesos'][-1] += factor_aprendizaje * neurona['delta']


def zerar_salidas(red):
    for camada in red:
        for neurona in camada:
            neurona['salida'] = 0


MSE = [] 
epocas = []

def entrenar_red(red, set_entrenamiento, factor_aprendizaje, momentum, epochs, COTA_ERROR):

	espacio_latente = []
	count_epochs = 0
	Error_w = 1

	while (Error_w > COTA_ERROR):

		Error_w = 0 #Funcion de costo E(W)
		espacio_latente = []
		
		for row in set_entrenamiento:

			first_example = True

			temporal_delta = [neurona['delta'] \
                for capa in red for neurona in capa] \
                if not first_example else None

			#Forward propagation
			salida, el = propagacion(red, row)
			espacio_latente.append(el)
			salida_deseada = row

			#Error entre la salida deseada e la salida calculada (Diap : 11/49)
			Error_w += ((np.linalg.norm(np.array(salida_deseada) - np.array(salida)))**2)*0.5

			backtracking(red, salida_deseada)
			actualizar_pesos(red, row, factor_aprendizaje, momentum, temporal_delta)
			zerar_salidas(red)
			first_example = False

		print('[*]epocas={0}, aprendizaje={1}, error={2}'.format(count_epochs, factor_aprendizaje, Error_w))

		#MSE.append(Error_w/len(set_entrenamiento))
		MSE.append(Error_w)
		epocas.append(count_epochs)


		count_epochs += 1
		if count_epochs == epochs:
			break

	return espacio_latente, MSE, epocas

def predict(red, entrada):
	#propagacion retorna : inputs y espacio_latente
	outputs,latente = propagacion(red, entrada)
	return outputs, latente


def main():

	num_entradas  = len(dataset_TRAIN2[0]) - 1 #35
	num_salidas   = 1
	capas_escondidas   = [2]
	COTA_ERROR         = 0.002 
	factor_aprendizaje = 0.001
	epochs             = 20000

	red = crear_red(num_entradas, capas_escondidas, num_salidas)
	entrenar_red(red, dataset_TRAIN2, factor_aprendizaje, epochs, COTA_ERROR)

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
