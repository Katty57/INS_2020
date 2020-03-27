import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
    
def plot_show(epochs, mae, loss, val_mae, val_loss):
    print(len(mae), len(loss), len(val_mae), len(val_loss))
    plt.plot(epochs, loss, 'm', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot(epochs, mae, 'm', label='Training mean absolute error')
    plt.plot(epochs, val_mae, 'b', label='Validation mean absolute error')
    plt.title('Training and validation mean absolute error')
    plt.xlabel('Epochs')
    plt.ylabel('Mean absolute error')
    plt.legend()
    plt.show()

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(test_targets)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 2
num_val_samples = len(train_data) // k
num_epochs = 350
all_scores = []
arr_loss = []
arr_mae = []
arr_val_loss = []
arr_v_al_mae = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, validation_data = (val_data, val_targets))
    loss = history.history['loss']
    arr_loss.append(loss)
    mae = history.history['mae']
    arr_mae.append(mae)
    val_loss = history.history['val_loss']
    arr_val_loss.append(val_loss)
    v_al_mae = history.history['val_mae']
    arr_v_al_mae.append(v_al_mae)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    plot_show(range(1, num_epochs + 1), mae, loss, v_al_mae, val_loss)
plot_show(range(1, num_epochs + 1), np.mean(arr_mae, axis = 0), np.mean(arr_loss, axis = 0), np.mean(arr_v_al_mae, axis = 0), np.mean(arr_val_loss, axis = 0))
print(np.mean(all_scores))
