import numpy as np
from sparse_layer import Sparse
import keras
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, InputLayer, Activation

def decode_chromosome(element, n):
    final = [element[i * n:(i + 1) * n] for i in range((len(element) + n - 1) // n)]
    return np.array(final)

def extend_chromosome(element, rep):
    longitud = len(element)
    final = []

    for i in range(longitud):
        content = element[i]
        repeated_content = [content] * rep 
        final.append(repeated_content)

    # devolvemos el vector
    flat_list = []
    for sublist in final:
        for item in sublist:
            flat_list.append(item)

    return flat_list


def build_model(connections, shape,num_classes,bias,num_layers, no_compile=False):
    #print("El bias es... ", bias)
    sparse = Sparse(adjacency_mat=connections, activation = "relu", use_bias = bias)

    model = Sequential()
    model.add(InputLayer(input_shape=shape))

    for i in range(num_layers[0]):
        if (i+1) == num_layers[1]:
            model.add(sparse)
        else:
            model.add(Dense(512,activation='relu'))

    #model.add(Dense(512, activation='relu', input_shape= shape))
    #model.add(sparse)
    model.add(Dense(num_classes))
    model.add(Activation("softmax")) ##  

    #model.add(Dense(num_classes, activation='softmax'))

    opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary(), flush = True)

    return model

def build_reference_model(shape,num_classes,drop,bias,options_layers_ref):
    #print("Creando modelo")
    print("El bias es... ", bias)
    model = Sequential()

    #model.add(InputLayer(input_shape=shape))
    model.add(Dense(options_layers_ref[1], activation='relu', input_shape=shape)) # antes 512

    #if drop >= 0:
    #    if drop == 1:
    #        model.add(Dropout(0.5))
    
    if options_layers_ref[0] > 1:
        for i in range(options_layers_ref[0]-1):    
            model.add(Dense(options_layers_ref[i+2], activation='relu', use_bias = bias))
    

    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    #model.add(Dense(num_classes, activation='softmax'))

    opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary(), flush = True)

    return model


def build_model_full(first_connections, second_connections, shape,num_classes,bias):
    #print("El bias es... ", bias)

    first_sparse = Sparse(adjacency_mat=first_connections, activation = "relu", use_bias = bias)
    second_sparse = Sparse(adjacency_mat=second_connections, activation = "relu", use_bias = bias)

    model = Sequential()
    model.add(InputLayer(input_shape=shape))
    #model.add(Dense(512, activation='relu', input_shape= shape))
    model.add(first_sparse)
    model.add(second_sparse)
    
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    #model.add(Dense(num_classes, activation='softmax'))
    
    opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary(), flush = True)

    return model


#from keras.utils import to_categorical
#import keras.backend as K
#num_samples = 100
#X = np.random.rand(num_samples, 10)
#y = np.random.randint(low=0, high=2, size=num_samples)
#y = to_categorical(y, num_classes=4)


#model = Sequential()
#model.add(InputLayer(input_shape=(10,)))
#model.add(Dense(4))
#model.add(Activation("softmax")) 

#opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#print(model.summary(), flush = True)

#model.fit(X, y, epochs=5, batch_size=10)


############## test

#layer_name = 'dense_1'
#intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

#X_test_prueba = np.random.rand(5, 10)
#intermediate_output = intermediate_layer_model.predict(X_test_prueba)
#print(intermediate_output)

#func = K.function([model.get_layer(index=0).input], model.get_layer(index=0).output)
#layerOutput = func([input_data])  # input_data is a numpy array
#print(layerOutput)
#for layer in model.layers:
#    weights = layer.get_weights()
#    print(weights)
#    print("----------------")

