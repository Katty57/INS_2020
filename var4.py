import gens
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def random_line(img_size=50):
    p = np.random.random([1])
    if p < 0.5:
        return gens.gen_v_line(img_size)
    else:
        return gens.gen_h_line(img_size)
        
def getData(size=500, img_size=50):
    x, y = gen_data(size, img_size)
    x, y = shuffle(x, y)
    return train_test_split(x, y, test_size = 0.2, random_state = 42)


def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1

    label_c1 = np.full([c1, 1], 'Cross')
    data_c1 = np.array([gens.gen_cross(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Line')
    data_c2 = np.array([random_line(img_size) for i in range(c2)])

    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))

    return data, label

batch_size = 23
num_classes = 2
epochs = 17

# input image dimensions
img_rows, img_cols = 50, 50

# the data, split between train and test sets
x_train, x_test, y_train, y_test = getData()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
    
encoder = LabelEncoder()
y_trn = encoder.fit_transform(y_train)
y_tst = encoder.fit_transform(y_test)

# convert class vectors to binary class matrices
y_trn = keras.utils.to_categorical(y_trn, num_classes)
y_tst = keras.utils.to_categorical(y_tst, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit(x_train, y_trn,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_tst))

score = model.evaluate(x_train, y_trn, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate(x_test, y_tst, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
