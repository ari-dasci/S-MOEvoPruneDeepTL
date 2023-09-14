from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from my_operators import UniformCrossover
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
#from visualization.chord_plot import *
from evoprune import EvoPruneDeepTL
from collections import Counter
########################################################################


from fitness import Fitness, Optimization_type
import sys
import time
from utility import read_dataset, read_dataset_ood
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="EvoPruneDeepTL", add_help=True)
parser.add_argument("--dataset", help="Config file for dataset", default="")
parser.add_argument("--configGA", help="Config for the GA", default="configGA.csv")
parser.add_argument("--run", help="Iteration number of EvoPruneDeepTL", type=int)
parser.add_argument("--extractor",help="Extractor", default="resnet50")

args = parser.parse_args()

file_dataset = args.dataset
file_params_ga = args.configGA
iteracion = args.run
extractor = args.extractor
#from_dir = args.features

#LECTURA DE DATOS
# Configuracion del dataset
# aqui leemos todo lo necesario para leer los datos

a0 = time.time()

file = open(file_dataset,"r")
datos, dimension = read_dataset(file,extractor)



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
#file_params_ga = sys.argv[2]
f = open(file_params_ga,"r")
config = f.readline().split(",")


# size de la poblacion
tam_pop = int(config[0])

# tipo: first = FS, second = pruning
tipo = str(config[1])

# input
input_size = int(config[2])

# priemra layer
primera = int(config[3])

# segunda layer (si hubiese)
segunda = int(config[4])

# max evaluaciones
max_evals = int(config[5])

# neuronas = 0, conexiones = 1
type_model = int(config[6])

# info sobre las capas
how_many = int(config[7])
which_sparse = int(config[8])
both_sparse = int(config[9])

info_layers = [how_many, which_sparse, both_sparse]


print("Configuracion")
print(config)

a_guardar = str(tam_pop)+str(tipo)+str(input_size) + "-" + str(extractor) + "-" + str(iteracion)
print(a_guardar)

optimization_type = Optimization_type.NEURONS  # no actualizamos la longitud de cromosoma pues es 512 para la primera capa y tambien para el GA
longitud_cromosoma = -1

if tipo == "second":
    if how_many == 1:
        longitud_cromosoma = primera
    else:
        if which_sparse == 1:
            longitud_cromosoma = primera
        elif which_sparse == 2:
            if both_sparse == 0:
                longitud_cromosoma = segunda
            else:
                longitud_cromosoma = primera + segunda
else:
    longitud_cromosoma = input_size

fitness = Fitness(input_size=input_size,first_layer=primera,second_layer=segunda,data=datos, layer=tipo, num_layers=info_layers,optimType=optimization_type, extractor=extractor, ood_data=data_ood, algorithm_features=["SPEA2",iteracion])

problem = EvoPruneDeepTL(number_of_bits=longitud_cromosoma,method=fitness,cadena=a_guardar)

print("Empezando a crear el NSGA-II...")
print("con longitud de cromosoma " + str(longitud_cromosoma))

algorithm = NSGAII(problem=problem,
                   population_size=tam_pop,
                   offspring_population_size=2,
                   mutation=BitFlipMutation(probability=1.0 / longitud_cromosoma),
                   crossover=UniformCrossover(),
                   termination_criterion=StoppingByEvaluations(max_evaluations=max_evals))

print("Tiempo antes de empezar el algoritmo")
print(str(time.time()-a0))
start_time = time.time()
algorithm.run()
solution = algorithm.get_result()
print("Tiempo transcurrido: ", str(time.time()-start_time))


#######################################
#print(type(solution))
#print(len(solution))
front = get_non_dominated_solutions(algorithm.get_result())
print(len(front))

guardar = datos[0] + str(iteracion)
print_function_values_to_file(front, 'FUN.NSGAII.'+guardar)
print_variables_to_file(front, 'VAR.NSGAII.'+guardar)
