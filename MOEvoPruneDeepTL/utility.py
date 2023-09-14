import numpy as np
from numpy import array
import os, sys
import glob
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.applications.densenet import preprocess_input as preprocess_input_densenet
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.datasets import mnist,cifar10, fashion_mnist
from keras.utils import np_utils

#from skimage.transform import resize

def read_dataset_ood(data,height=None, width=None,ejemplos=None):
    config = data.readline().split(",")

    name = str(config[0])
    dir_train = str(config[1])
    num_classes = int(config[6])
    length = count_files(dir_train)

    examples_per_class = ejemplos // num_classes
    sobrantes = ejemplos % num_classes
    elementos_finales_clase = [examples_per_class] * num_classes

    for i in range(sobrantes):
        elementos_finales_clase[i] += 1
    
    len_dataset = count_files(dir_train)
    #print(len_dataset)
    generador = IDG(preprocessing_function=preprocess_input_resnet)
    iterador = generador.flow_from_directory(dir_train, batch_size=len_dataset,target_size=(height,width),shuffle=False)

    return iterador,elementos_finales_clase


def read_dataset(data,extractor, height=None,width=None):

    if extractor == "resnet50":
        generador = IDG(preprocessing_function = preprocess_input_resnet)
        generadorTest = IDG(preprocessing_function =  preprocess_input_resnet)
    elif extractor == "densenet":
        generador = IDG(preprocessing_function = preprocess_input_densenet)
        generadorTest = IDG(preprocessing_function =  preprocess_input_densenet)
    else:
        generador = IDG(preprocessing_function = preprocess_input_vgg19)
        generadorTest = IDG(preprocessing_function =  preprocess_input_vgg19)

    # lectura de los datos
    config = data.readline().split(",")
    # en cofing tenemos todo lo necesario para leer el dataset
    print(config)
    name = str(config[0])
    #dir_train = "./"+str(config[1])
    #dir_test = "./"+str(config[2])
    dir_train = str(config[1])
    dir_test = str(config[2])
    #len_train = int(config[3])
    #len_test = int(config[4])

    #len_train = len([f for f in dir_train if os.path.isfile(os.path.join(dir_train
    #len_test = 
    num_folds = int(config[3])

    if height == None:
        height = int(config[4])

    if width == None:
        width = int(config[5])

    num_classes = int(config[6])
    it = []
    it_val = []

    print(dir_train)
    print(dir_test)

    if dir_train != "-":
        if num_folds == 1:
            len_train = count_files(dir_train)
            print(dir_train)
            iterator_train = generador.flow_from_directory(dir_train,batch_size=len_train,target_size=(height,width),shuffle=True)
        else:
            for i in range(num_folds):
                #leemos su ruta final
                dir_final = dir_train+str(i)+"/"
                len_train = count_files(dir_final)
                print(len_train)
                print(dir_final)
                iterator = generador.flow_from_directory(dir_final,batch_size=len_train,target_size=(height,width),shuffle=True)
                it.append(iterator)

    if dir_test != "-":
        if num_folds == 1:
            len_test = count_files(dir_test)  
            iterator_test = generadorTest.flow_from_directory(dir_test,batch_size=len_test,target_size=(height,width),shuffle=False)
        else:
            for i in range(num_folds):
                dir_final = dir_test+str(i)+"/"
                len_test = count_files(dir_final)
                print(len_test)
                print(dir_final)
                iterator_val = generadorTest.flow_from_directory(dir_final,batch_size=len_test,target_size=(height,width),shuffle=False)
                it_val.append(iterator_val)

    if num_folds != 1:
        iterator_train = it
        iterator_test = it_val

    print(type(iterator_train))

    return [name,iterator_train,iterator_test,len_train,len_test,num_classes], [height,width]

def count_files(folder):
    total = 0

    for root, dirs, files in os.walk(folder):
        total += len(files)

    return total
def read_population(file):
    population = []
    accuracies = []

    f = open(file,"r")
    count = 0
    line = f.readline().split(",")

    while True:
        population.append(line[0])
        accuracies.append(line[1])
        count += 1

        line = f.readline()

        if not line:
            break

        line = line.split(",")

    f.close()
    #print(count)
    return population,accuracies

def get_best_elements(num,population,accuracies):
    best_pop = []
    best_accs = []

    indexes = sorted(range(len(accuracies)), key=lambda i: accuracies[i],reverse=True)[:num]
    #print(indexes)

    for i in range(len(indexes)):
        best_accs.append(accuracies[indexes[i]])
        best_pop.append(population[indexes[i]])

    #print(best_accs)
    return best_pop,best_accs,indexes


def get_position_element(num, population,element,num_neuron):
    new_population = []
    sliced_pop = slice_into(num,population,element) # conseguimos los grupos de N neuronas de cada ellos y ahora tenemos que coger el elemento de pos num_neuron

    for i in range(len(sliced_pop)):
        new_population.append(sliced_pop[i][num_neuron])

    return new_population


def slice_into(num, population, element):
    new_population = []

    position_ini = element * num
    position_fin = element * num + num

    for i in range(len(population)):
        sliced = population[i][position_ini:position_fin]
        new_population.append(sliced)

    return new_population

#comprobar que sigue funcionando
def read_matrices(directory):
    num_a_leer = len(glob.glob(directory+"*"))
    carpetas = glob.glob(directory+"*")
    carpetas.sort(key=os.path.getmtime)
    matrices_finales = []


    for i in range(num_a_leer):
        directorio = carpetas[i] + "/*" #aqui tenemos 0,1,...,25
        matrices_a_leer = glob.glob(directorio)
        matrices_a_leer.sort(key=os.path.getmtime)

        for j in range(len(matrices_a_leer)): # vemos todos los directorios de dichaa iteracion
            matrices = []
            carpetas_matrices = matrices_a_leer[j] + "/*" # aqui tenemos la ruta alg../iteracion/elemento

            a_leer = glob.glob(carpetas_matrices) # numero de elementos a leer (las 5 matrices


            for k in range(len(a_leer)):
                #print("Matriz " + str(a_leer[k]))
                matrices.append(np.load(a_leer[k]))

            # matriz conjunta
            matriz = 0
            for k in range(len(matrices)):
                matriz += matrices[k]

            matrices_finales.append(matriz)


    # devuelve todas las matrices de confusion agregadas
    #print(matrices_finales)
    return matrices_finales

def calc_precision(matriz):
    precision = 0
    numerador = matriz[1][1]
    denominador = matriz[1][1] + matriz[0][1]

    if denominador == 0:
        precision = -1
    else:
        precision = numerador / denominador

    return precision

def calc_recall(matriz):
    recall = 0
    numerador = matriz[1][1]
    denominador = matriz[1][1] + matriz[1][0]

    #print(matriz[1][0])

    if denominador == 0:
        recall = -1
    else:
        recall = numerador / denominador

    return recall

# F1 = 2 * (precision * recall) / (precision + recall)
def calc_f1(matrices):
    list_f1 = []

    for i in range(len(matrices)): # para cada configuracion de red
        f1_per_class = []

        for j in range(len(matrices[i])): # para cada clase
            matriz = matrices[i][j] # esto tiene una matriz de confusion para la clase i-esima
            rec = calc_recall(matriz)
            prec = calc_precision(matriz)

            if (prec + rec) != 0:
                f1 = 2 * (prec * rec) / (prec + rec)
            else:
                f1 = -1

            f1_per_class.append(f1)

        list_f1.append(f1_per_class)

    return list_f1


def calc_f1_macro(f1s):
    macro_f1 = []

    for i in range(len(f1s)):
        macro_f1.append(np.array(f1s[i]).mean())

    return macro_f1

# devolvemos el número de clases con f1 superior al umbral y las posiciones
def check_f1_upper_than(threshold,f1s):
    lista_final = []

    for i in range(len(f1s)):
        list_f1 = f1s[i] # tomamos para cada configuracion su lista con las 14 f1s, una por clase
        diccionario = dict()
        posiciones = []
        numero_superadas = 0

        for j in range(len(list_f1)):  #vemos qué clases tienen f1 superior a ese umbral.
            if list_f1[j] >= threshold:
                #print("F1 :" + str(list_f1[j]))
                numero_superadas +=1
                posiciones.append(j)

        diccionario['total'] = numero_superadas
        diccionario['posiciones'] = posiciones
        lista_final.append(diccionario)

    return lista_final
