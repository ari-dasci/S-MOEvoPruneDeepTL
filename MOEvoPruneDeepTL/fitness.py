#!/usr/bin/python3
from __future__ import division

import os
import numpy as np
from numpy.random import seed
#seed(1)
from tensorflow import set_random_seed
import keras

print(keras.__version__)
#set_random_seed(2)
#from keras.optimizers import SGD
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.applications.densenet import preprocess_input as preprocess_input_densenet
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.models import Sequential,Model,model_from_json, load_model, save_model
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.vgg19 import VGG19
from keras import callbacks
from build_model import build_model, decode_chromosome, build_reference_model, build_model_full
from importlib import reload
from enum import Enum
import time
import random
import sys
import ssl
from ood import ood_detection
import statistics
from sparse_layer import Sparse

ssl._create_default_https_context = ssl._create_unverified_context
#from sklearn.decomposition import PCA

def set_keras_backend(backend):
    if K.backend() != backend:
        print(K.backend())
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("tensorflow")
keras.backend.set_image_data_format("channels_first")
keras.backend.set_image_dim_ordering('tf')
session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
K.set_session(session)


def getFeatures(model, images, batch_size):
    return model.predict(images, batch_size=batch_size)


def next_raw_data(raw_data):
    list_raw_images = []

    for k in range(len(raw_data)):
        iterador_d = raw_data[k]
        xdata, ydata = iterador_d.next()
        list_raw_images.append([xdata,ydata])

    return list_raw_images

class Optimization_type(Enum):
    "Optimization type of Network: By neurons or by connections"
    NEURONS = 1
    CONNECTIONS = 2

class Fitness():
    def __init__(self, input_size,first_layer, second_layer, data,layer, num_layers,bias=False,optimType=Optimization_type.NEURONS,options_layers_ref=None,extractor="resnet50", ood_data=None, algorithm_features=["NSGA",-1]):
        self.input_size = input_size
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.data = data
        self.optimType = optimType
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.extractor = extractor
        self.ood_data = ood_data[0]
        self.examples_ood = ood_data[1]
        self.alg_fet = algorithm_features
        self.ood_data = next_raw_data(self.ood_data) # es una lista de la forma [datos,etiquetas]
        #self.predict = predict

        if self.layer == None:
            self.options_layers_ref = options_layers_ref

    def fitness(self, solution):
        #vemos el numero de 1
        neuronas_activas = solution.count(1) / len(solution)

        if self.layer == "second": # revisar todo esto para que sea fs o pruning
            if self.num_layers[0] == 1:
                num = self.first_layer
            else:
                if self.num_layers[1] == 1:
                    num = self.first_layer
                else:
                    num = self.second_layer
        elif self.layer == "first":
            num = self.input_size
    
        if self.num_layers[2] == 0 : # optimizacion de una capa
            if self.optimType == Optimization_type.NEURONS: 
                print("Solution")
                #print(solution)
                print(len(solution))

                if self.layer == "second": # TODO: revisar esto
                    if self.num_layers[0] == 1: # si es 1 capa 
                        print("Pruning 1 capa")
                        assert len(solution) == self.first_layer # como tengo 512, los repito 2048
                        solution = np.repeat(solution, self.input_size)
                    else: # si tenemos dos capas, vemos donde esta la sparse
                        if self.num_layers[1] == 1: # capa sparse en la primera
                            print("Pruning dos capas con sparse en la primera")
                            assert len(solution) == self.first_layer # como tengo 512, los repito 2048
                            solution = np.repeat(solution, self.input_size)
                        else: # capa sparse en la segunda
                            print("Pruning dos capas con sparse en la segunda")
                            assert len(solution) == self.second_layer # como tengo 512, los repito 512
                            solution = np.repeat(solution, self.first_layer)
                else: # fs
                    print("Feature Selection")
                    assert len(solution) == self.input_size # como tengo 2048 elementos, los repito 512
                    solution = np.repeat(solution, self.first_layer)
                    
                #solution = np.repeat(solution, num)#.tolist()
                print("Solution repetida")
                print(solution)
                print(len(solution))
                print("Num unos: ", str(list(solution).count(1)))
            else: # TODO revisar esto
                print("Para conexiones")

                if self.num_layers[0] == 1: # si tenemos una capa solo
                    assert len(solution) == self.first_layer*self.input_size
                else:
                    if self.num_layers[1] == 1:
                        assert len(solution) == self.first_layer*self.input_size
                    else:
                        assert len(solution) == self.first_layer*self.second_layer
        else: # both layers 
            if self.optimType == Optimization_type.NEURONS:
                assert len(solution[int(len(solution)/2):]) == self.first_layer and len(solution[:int(len(solution)/2)]) == self.second_layer
                #solution = np.repeat(solution, )#.tolist()
                print("Solution repetida")
                #print(solution)
                print(len(solution))
                print("Num unos: ", str(list(solution).count(1)))
            else:
                print("Para conexiones") # ESTO hay  que comprobar para conexiones
                assert len(solution) == (self.input_size*self.first_layer + self.first_layer * self.second_layer)
        
        return self._fitness(sol=solution,neuronas=neuronas_activas)


    def fitness_dense(self, drop=False):
        return self._fitness(sol=None, drop=drop)


    def _fitness(self, sol=None, drop=False,neuronas=None):
        #fijamos las semillas al comienzo de la función fitness

        a1 = time.time()
        seed(1)
        set_random_seed(2)

        accs = 0
        accs_train = 0
        auroc_fold = 0.0
        auroc = 0.0
        auroc_f = 0.0

        num_folds = len(self.data[1])
        print(num_folds)

        num_classes = self.data[5]
        print("Clases: " + str(num_classes))
        model = None

        for i in range(num_folds):
            model = None
            print("Set "+ str(i), flush = True)
            history = []

            if self.extractor == "resnet50":
                base_model = ResNet50(include_top=False, weights="imagenet", pooling="avg")
            elif self.extractor == "densenet":
                base_model = DenseNet121(include_top=False, weights="imagenet", pooling="avg")
            else:
                base_model = VGG19(include_top=False, weights="imagenet", pooling="avg")

            if num_folds != 1:
                x,y = self.data[1][i].next()
                x_ = getFeatures(base_model, x,32)

                xVal,yVal = self.data[2][i].next()
                xVal_= getFeatures(base_model, xVal, 32)
            else: 
                x,y = self.data[1].next()
                xVal, yVal = self.data[2].next()

                x_ = getFeatures(base_model, x, 32)
                xVal_= getFeatures(base_model, xVal,32)

            del x
            del xVal

            print("Train ", x_.shape, y.shape, flush = "True")
            print("Test ", xVal_.shape, yVal.shape, flush = "True")


            if sol is None:
                model = build_reference_model(x_.shape[1:],num_classes, drop,self.bias,self.options_layers_ref)
            else:
                if self.num_layers[2] == 0: # optimizacion de una capa # comprobar tb aqui el flag activo o no de consecutivo

                    if self.num_layers[0] == 1:
                        if self.layer == "first": # fs
                            matrix_connections = decode_chromosome(sol, self.first_layer)
                        elif self.layer == "second":
                            print("En transpuesta")
                            matrix_connections = decode_chromosome(sol, self.input_size) # creo que aqui es input_size
                            matrix_connections = matrix_connections.transpose()
                    else: # tenemos dos capas y siempre es second
                        if self.num_layers[1] == 1: # la sparse está en la primera
                            matrix_connections = decode_chromosome(sol, self.input_size) #creo que aqui es input_size
                            matrix_connections = matrix_connections.transpose()
                        else: # la sparse está en la segunda
                            matrix_connections = decode_chromosome(sol, self.first_layer)
                            matrix_connections = matrix_connections.transpose()


                    print("Configuracion")
                    print(matrix_connections.shape, flush = True)
                    print(matrix_connections, flush = True)

                    model = build_model(matrix_connections, x_.shape[1:], num_classes,self.bias, self.num_layers)
                else: # optimizacion de dos capas
                    # comprobar el de conexiones que hay que añadir los if-else

                    if self.optimType == Optimization_type.NEURONS:
                        first_sparse = sol[:int(len(sol)/2)]
                        second_sparse = sol[int(len(sol)/2):]

                        print(len(first_sparse))
                        print(len(second_sparse))

                        first_sparse = np.repeat(first_sparse,self.input_size)
                        second_sparse = np.repeat(second_sparse,self.first_layer)
                    else:
                        first_spase = sol[:int(self.input_size*self.first_layer)]
                        second_sparse = sol[int(self.first_layer*self.second_layer):]


                    first_matrix_connections = decode_chromosome(first_sparse, self.input_size)
                    first_matrix_connections = first_matrix_connections.transpose()

                    second_matrix_connections = decode_chromosome(second_sparse, self.first_layer)
                    second_matrix_connections = second_matrix_connections.transpose()                


                    print("Configuracion")
                    print(first_matrix_connections.shape, flush = True)
                    print(second_matrix_connections.shape, flush = True)
                    print(first_matrix_connections, flush = True)
                    print(second_matrix_connections, flush = True)
                    print(x_.shape[1:],flush=True)
                    model = build_model_full(first_matrix_connections,second_matrix_connections ,x_.shape[1:], num_classes,self.bias)

            callbacks_array = [callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=10, verbose=1, mode="min", restore_best_weights=True)]
            print("COMENZADO A ENTRENAR", flush=True)
            a0 = time.time()
            history = model.fit(x_, y, epochs=600, batch_size=32, verbose=0, callbacks=callbacks_array)
            a1 = time.time()
            print("TIEMPO DE ENTRENAMIENTO", flush=True)
            print(str(a1-a0), flush =True)
            ## AQUI YA TENGO EL MODELO ENTRENADO
            ## copiamos el modelo, pero solamente con la salida como los logits antes del softmax
            print("COMENZANDO OOD", flush=True)
            auroc_f = self._calc_ood_detection(model,xVal_)
            a2 = time.time()
            print("TIEMPO DE OOD", flush=True)
            print(str(a2-a1), flush=True)

            print("Fit "+str(set)+
                    " acc:" + str(history.history['acc'][-1])[:6] +
                    " loss:"+ str(history.history['loss'][-1])[:6],flush = True)

            print("INFERENCE", flush=True)
            [test_loss, acc_test] = model.evaluate(xVal_,yVal)
            a3 = time.time()
            print(str(a3-a2),flush=True)

            print(test_loss, acc_test)
            accs += acc_test
            accs_train += history.history['acc'][-1]
            auroc_fold += auroc_f

            self.save_model(model,i,acc_test,auroc_f,neuronas)
            #else:

#
#                model = load_model("models/"+nombre_modelo, custom_objects={'Sparse': Sparse})
#                model.summary()
#
#                [test_loss, acc_test] = model.evaluate(xVal_,yVal)
#                print("Evaluate del modelo")
#                print(test_loss,acc_test)
#
#                print("OOD para el modelo cargado")
#                auroc_f = self._calc_ood_detection(model,xVal_)
#
#                return 1-acc_test, 1-auroc_f


            #[test_loss, acc_test] = model.evaluate(xVal_,yVal)
            #print(test_loss, acc_test)
    

            # update de auroc y accs
            #accs += acc_test
            #accs_train += history.history['acc'][-1]
            #auroc_fold += auroc_f

            # guardado del modelo
            #self.save_model(model,i,acc_test,auroc_f,neuronas)
        
            K.clear_session()
            

        print("Accuracy cruzada: "+str(accs/num_folds), flush = True)
        acc = accs/num_folds
        print("Accuracy TRAIN: " + str(accs_train/num_folds), flush = True)
        print("AUROC cruzada: " + str(auroc_fold/num_folds), flush = True)
        auroc = auroc_fold/num_folds

        # se fija semilla a un valor aleatorio
        new_seed = random.randrange(2**32-1)
        print("Nueva seed " + str(new_seed)) 
        seed(new_seed)
        return acc, auroc

    def _calc_ood_detection(self,model,test_features):
        #print("Empezando a calcular el ood... ")
        layer_name='dense_1'

        # tomamos el modelo intermedio cuyos outputs son los logits para el test y para el ood
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

        # calculamos ahora los logits de test
        logits_test = intermediate_layer_model.predict(test_features)
        #print("SHAPES LOGITS TEST")
        #print(logits_test.shape)

        # ahora hacemos lo mismo para calcular el logits_ood para cada conjunto de datos
        logits_ood = []
        partial_auroc_per_dataset = 0.0
        aurocs = []
        auroc_dataset = 0.0

        # para cada iterador (para cada base de datos, extraemos sus ejemplos y calculamos los logits
        
        b8 = time.time()
        for d in range(len(self.ood_data)):
            # tomamos el iterador asociado a ese dataset y el numero de ejemplos para cada clase
            ejemplos_d = self.examples_ood[d]
            # cargamos el modelo y le pasamos solamente las imágenes que queremos
            #base_model_ood = ResNet50(include_top=False, weights="imagenet", pooling="avg")
            xdata = self.ood_data[d][0]
            ydata = self.ood_data[d][1]
            idx_images = self._process_ood_data(ydata, ejemplos_d)
            ood_images = xdata[idx_images]
            #b1 = time.time()

            #print("Tiempo en procesar unos " + str(b1-b0), flush=True)
            #print("NUM IMAGENES")
            #print(ood_images.shape)

            base_model_ood = ResNet50(include_top=False, weights="imagenet", pooling="avg")
            ood_features = getFeatures(base_model_ood, ood_images, 32)

            #b11 = time.time()
            #print("Tiempo en obtener features " + str(b11-b1), flush=True)

            # calculo los logits_ood para ese dataset
            logits_ood_d = intermediate_layer_model.predict(ood_features)

            #b12 = time.time()
            #print("Tiempo en calcular logits " + str(b12-b11), flush=True)
            logits_ood.append(logits_ood_d)

            #print("SHAPES LOGITS OOD")
            #print(logits_ood_d.shape)


        # agregamos todos los logits_ood y los mandamos para calcular el logits total de ood
        logits_ood_final = np.vstack(logits_ood)
        #print("SHAPES LOGITS OOD FINALES")
        #print(logits_ood_final.shape)

        #calculo el auc
        #b2 = time.time()

        #print("BUCLE OOD " + str(b2-b8), flush=True)

        auroc_dataset = ood_detection(logits_test, logits_ood_final, type_ood="odin")

        return auroc_dataset

    def _process_ood_data(self,ydata, num_ejemplos):
        chosen_idx = []
        ref = ydata[0]
        l = 0
        actual = 0
        copy_ejemplos = num_ejemplos[:]

        while (l < ydata.shape[0]):
            if ((ydata[l] == ref).all() and copy_ejemplos[actual] > 0): # podemos añadir otro ejemplo mas
                chosen_idx.append(l)
                copy_ejemplos[actual] -= 1 # quitamos un ejemplo
            elif not (ydata[l]==ref).all(): #cuando nos quedamos sin ejemplos de esa clase, saltamos hasta la siguiente (realmente es seguir avanzando en l sobre la lista y esperar a que cambien
                actual += 1
                ref = ydata[l]
                chosen_idx.append(l)
                copy_ejemplos[actual] -= 1

            l += 1

        return chosen_idx

    def save_model(self,modelo,fold,acc,ood_value,active_neur):
        numero_folds = len(self.data[1])
        ruta = "./models/" + self.alg_fet[0] + "-" + self.get_problem_name() + "-" + str(self.alg_fet[1]) + "-"

        if numero_folds > 1:
            ruta += str(fold) + "-"

        ruta += str(1-acc) + "-" + str(active_neur) + "-" + str(1-ood_value)

        modelo.save_weights(ruta+".h5")
        print("Saved model")
        
    def load_predict_model(self,ruta,modelo):
        modelo.load_weights("models/"+ruta)
        print("Loaded model from disk")

        num_folds = len(self.data[1])
        print(num_folds)
        num_classes = self.data[5]
        print("Clases: " + str(num_classes))

        for i in range(num_folds):
            print("Set "+ str(i), flush = True)
            base_model = ResNet50(include_top=False, weights="imagenet", pooling="avg")

            if num_folds != 1:
                x,y = self.data[1][i].next()
                x_ = getFeatures(base_model, x,32)
                xVal,yVal = self.data[2][i].next()
                xVal_= getFeatures(base_model, xVal, 32)
            else:
                x,y = self.data[1].next()
                xVal, yVal = self.data[2].next()
                x_ = getFeatures(base_model, x, 32)
                xVal_= getFeatures(base_model, xVal,32)

            del x
            del xVal

            print("Train ", x_.shape, y.shape, flush = "True")
            print("Test ", xVal_.shape, yVal.shape, flush = "True")


            #opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
            #modelo.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            #print(modelo.summary(), flush = True)

            print("Evaluate del modelo")
            [test_loss, acc_test] = modelo.evaluate(xVal_,yVal)
            print("Accuracy en TEST")
            print(acc_test)

            print("Calculamos OOD para el modelo cargado")
            auroc_f = self._calc_ood_detection(modelo,xVal_)

            print("AUROC")
            print(auroc_f)


    def get_problem_name(self):
        return self.data[0]
