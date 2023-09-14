from fitness import Fitness, Optimization_type
import sys
import time
from utility import read_dataset, read_dataset_ood
import numpy as np
import argparse
from os import listdir
from build_model import build_model, decode_chromosome
from sparse_layer import Sparse
from collections import Counter

parser = argparse.ArgumentParser(description="EvoPruneDeepTL", add_help=True)
parser.add_argument("--dataset", help="Config file for dataset", default="")

args = parser.parse_args()
file_dataset = args.dataset
##########################################################################################################################################################
#LECTURA DE DATOS
# Configuracion del dataset
# aqui leemos todo lo necesario para leer los datos

file = open(file_dataset,"r")
datos, dimension = read_dataset(file,extractor="resnet50")

#### LECTURA DE DATOS PARA OOD

files=["Corales", "Leaves", "PinturasFull", "RPS","Ojos","Plantas"]
datos_resto = []
ejemplos_resto = []

#files=["Plantas"]

# obtenemos el número de ejemplos de test del dataset a analizar
# vemos si tiene solo un fold o varios

if datos[0] != "SRSMAS" and datos[0] != "LEAVES":
    counter = Counter(datos[2].classes)
else:
    counter = Counter(datos[2][0].classes)

total_test = 0

for c in counter.items():
    total_test += c[1]

ejemplos_por_clase = total_test // (len(files)-1) 
restantes = total_test % (len(files)-1)
ejemplos_test = [ejemplos_por_clase] * (len(files)-1)

for i in range(restantes):
    ejemplos_test[i] += 1

contador=0

for f in files:
    if f not in file_dataset:
        cadena = "configDataset" + f + "OOD.csv"
        nueva_config = open(cadena,"r")
        datos_f, elementos_clase = read_dataset_ood(nueva_config,height=dimension[0], width=dimension[1], ejemplos = ejemplos_test[contador])
        datos_resto.append(datos_f)
        ejemplos_resto.append(elementos_clase)
        contador+=1

data_ood = [datos_resto,ejemplos_resto]


#siguiente_capa = int(config[3])
# ejecutamos el modelo genetico
# Parámetros para el GA
# size de la poblacion
tam_pop = 30
# tipo: first = FS, second = pruning
tipo = "first"
# input
input_size = 2048
# priemra layer
primera = 512
# segunda layer (si hubiese)
segunda = -1
# max evaluaciones
max_evals = 200
# neuronas = 0, conexiones = 1
type_model = 0
# info sobre las capas
how_many = 1
which_sparse = 1
both_sparse = 0
info_layers = [how_many, which_sparse, both_sparse]
iteracion=-1

optimization_type = Optimization_type.NEURONS  # no actualizamos la longitud de cromosoma pues es 512 para la primera capa y tambien para el GA
longitud_cromosoma = input_size

fitness = Fitness(input_size=input_size,first_layer=primera,second_layer=segunda,data=datos, layer=tipo, num_layers=info_layers,optimType=optimization_type, extractor="resnet50", ood_data=data_ood)
########################################################################################################################################################################################################

solution = [1]*2048
solution = np.repeat(solution,primera)
matrix_connections = decode_chromosome(solution,primera)

# indicar el número de clases para que funcione bien el modelo
esquema = build_model(connections = matrix_connections,shape = (2048,),num_classes = 3,bias=False,num_layers=info_layers)


print("Cargando los modelos ...")
files = listdir("./models")
files = [f for f in files if f.endswith(".h5")]
print(files)

#ruta = "models/"+files[0]
#print(ruta)

for f in files:
    fitness.load_predict_model(f,esquema)
