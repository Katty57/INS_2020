from keras.layers import Input, Embedding, LSTM, Dense, Flatten
from keras.models import Model, Sequential
import numpy as np
import csv
import matplotlib.pyplot as plt

def create_csv(filename, data):
    with open(filename, 'w', newline = '') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        try:
            for row in data:
                filewriter.writerow(row)
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(filename, row, e))

def generate_dataset(name, n):
    data = []
    targets = []
    for i in range(n):
        x = np.random.normal(3, 10)
        e = np.random.normal(0, 0.3)
        data.append((x**2+e, np.sin(x/2)+e, np.cos(2*x)+e, x-3+e, np.fabs(x)+e,(x**3)/4+e))
        targets.append([-x+e])
    create_csv(name + '_data.csv', data)
    create_csv(name + '_targets.csv', targets)
    return np.array(data), np.array(targets)

train_data, train_targets = generate_dataset('train', 300)
test_data, test_targets = generate_dataset('test', 60)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

main_input = Input(shape=(6,), name = 'main_input')
encoded = Dense(60, activation='relu')(main_input)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(6, activation='relu')(encoded)

encoded_input = Input(shape=(6,), name = 'encoded_input')
decoded = Dense(32, activation='relu')(encoded_input)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(16, activation='relu')(decoded)
decoded = Dense(6, name='decoded_layer')(decoded)

predicted = Dense(64, activation='relu')(encoded)
predicted = Dense(32, activation='relu')(predicted)
predicted = Dense(1, name='predicted_layer')(predicted)

encoded = Model(main_input, encoded, name='encoder')
decoded = Model(encoded_input, decoded, name='decoder')
predicted = Model(main_input, predicted, name='regression')

model = Sequential()
model.add(Dense(60, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
h = model.fit(train_data, train_targets, epochs=40, batch_size=10, verbose=1, validation_data=(test_data, test_targets))

#loss = h.history['loss']
#v_loss = h.history['val_loss']
#plt.plot(range(1, 41), loss, 'b', label='train')
#plt.plot(range(1, 41), v_loss, 'r', label='test')
#plt.title('loss')
#plt.ylabel('loss')
#plt.xlabel('epochs')
#plt.legend()
#plt.show()
#plt.clf()

predicted.compile(optimizer='adam', loss='mse', metrics=['mae'])
h = predicted.fit(train_data, train_targets,epochs=40, batch_size=10, verbose=1, validation_data=(test_data, test_targets))

encoded_data = encoded.predict(test_data)
decoded_data = decoded.predict(encoded_data)

#loss = h.history['loss']
#v_loss = h.history['val_loss']
#x = range(1, 41)
#plt.plot(x, loss, 'b', label='train')
#plt.plot(x, v_loss, 'r', label='test')
#plt.title('Loss')
#plt.ylabel('loss')
#plt.xlabel('epochs')
#plt.legend()
#plt.show()
#plt.clf()

create_csv('encoded.csv', encoded_data)
create_csv('decoded.csv', decoded_data)
create_csv('result.csv', predicted.predict(test_data))

decoded.save('decoder.h5')
encoded.save('encoder.h5')
predicted.save('model.h5')
