import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

def logic_operation(a, b, c):
    return (int)((a and b) or (a and c))
    
def naive_relu(x):
    return np.maximum(x, 0)
    
def naive_sigmoid(x):
    return 1/(1+np.exp(-x))
    
def get_layers(weights):
    layers = []
    for i in range(len(weights)-1):
        layers.append(naive_relu)
    layers.append(naive_sigmoid)
    return layers
    
    
def by_elem(res, weights):
    data = res.copy()
    layers = get_layers(weights)
    for i in range(len(weights)):
        step = np.zeros((len(data), len(weights[i][1])))
        for j in range(len(data)):
            for k in range(len(weights[i][1])):
                sum = 0
                for l in range(len(data[j])):
                    sum += (data[j][l] * weights[i][0][l][k])
                step[j][k] = layers[i](sum + weights[i][1][k])
        data = step
    return data
    
def tensor_oper(res, weights):
    data = res.copy()
    layers = get_layers(weights)
    for i in range(len(weights)):
        data = layers[i](np.dot(data, weights[i][0])+weights[i][1])
    return data

    
def get_res(dataset, model):
    weights = []
    for l in model.layers:
        weights.append(l.get_weights())
    print("MODEL")
    print(model.predict(dataset))
    print("BY_ELEM")
    print(by_elem(dataset, weights))
    print("TENSOR")
    print(tensor_oper(dataset, weights))
    

dataset = np.array([[0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1]])
model = Sequential()
model.add(Dense(7, activation = 'relu', input_shape = (3,)))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

operation_res = np.zeros((8,1))
for d in dataset:
    operation_res[d] = logic_operation(d[0], d[1], d[2])

print("BEFORE FITTING")
get_res(dataset, model)
print("AFTER FITTING")
model.fit(dataset, operation_res, epochs = 35, batch_size = 2)
get_res(dataset, model)
