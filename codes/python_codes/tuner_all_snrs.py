import tensorflow

# from codes.mat_to_hkl_SNR_jobs import SNR
# with tensorflow.device('/device:GPU:0'):
from tensorflow.keras.layers import Reshape, SimpleRNN, Dense, Conv1D, LSTM, Dropout, Conv2D, MaxPooling2D, \
    MaxPooling1D, Flatten, Embedding, Bidirectional, Input
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import sys
from windowing import windOwized, LDG
import numpy as np
import pickle

# seed = int(sys.argv[1])

seed = 1

load_dir = "./data/processed/"
tuner_model_dir = "./tuner_models/"

snr = int(sys.argv[1])  # dB
num_RF_chains = 6
string = "SNRs_"
snr_set = [snr]
for snr in snr_set:
    string += str(snr) + "_"

model_num = "automml_" + string + "RF" + str(num_RF_chains) + "_seed" + str(seed)
techs = {"w": [], "l": [], "g": []}
tech_labels = {"w": 0, "l": 1, "g": 2}

mcs = 64

size = 100_000  ##number of datapoints per SNR

if snr_set:
    size_per_snr = size // len(snr_set)
    for tech in techs:
        for snr in snr_set:
            file_name = "Waveforms_" + tech + "_" + str(mcs) + "_" + str(snr) + "_rf6.hkl"
            techs[tech] += [windOwized(load_dir, file_name, 710, 710, size_per_snr, num_RF_chains)[0]]
        techs[tech] = np.array(techs[tech])
        shape = np.shape(techs[tech])
        techs[tech] = techs[tech].reshape((shape[0] * shape[1], shape[2], shape[3]))
else:
    for tech in techs:
        file_name = "Waveforms_" + tech + "_" + str(mcs) + "_" + str(snr) + "_rf6.hkl"
        techs[tech] = windOwized(load_dir, file_name, 710, 710, size, 6)[0]
        techs[tech] = np.array(techs[tech])

min_waveforms = np.min([len(techs[tech]) for tech in techs])
labels = []  # labels
features = []  # features
for tech in techs:
    features.append(techs[tech][0:min_waveforms])
    labels.append(np.ones((size, 1)) * tech_labels[tech])
features = np.array(features)
f_shape = np.shape(features)
features = features.reshape((f_shape[0] * f_shape[1], f_shape[2], f_shape[3]))

labels = np.array(labels)
l_shape = np.shape(labels)
labels = labels.reshape((l_shape[0] * l_shape[1], l_shape[2]))

scaler = MinMaxScaler()
for i, x in enumerate(features):
    scaler = MinMaxScaler()
    scaler.fit(features[i])
    features[i] = scaler.transform(x)

labels = labels[:, 0]

labels = labels.reshape(-1)

# num_RF_chains=hp.get("#antennas")
X = features
del features
if num_RF_chains == 6:
    X = np.concatenate((X[:, :, 0:2, np.newaxis], X[:, :, 2:4, np.newaxis], X[:, :, 4:6, np.newaxis],
                        X[:, :, 6:8, np.newaxis], X[:, :, 8:10, np.newaxis], X[:, :, 10:12, np.newaxis]), axis=3)
elif num_RF_chains == 5:
    X = np.concatenate((X[:, :, 0:2, np.newaxis], X[:, :, 2:4, np.newaxis], X[:, :, 4:6, np.newaxis],
                        X[:, :, 6:8, np.newaxis], X[:, :, 8:10, np.newaxis]), axis=3)

elif num_RF_chains == 4:
    X = np.concatenate(
        (X[:, :, 0:2, np.newaxis], X[:, :, 2:4, np.newaxis], X[:, :, 4:6, np.newaxis], X[:, :, 6:8, np.newaxis]),
        axis=3)
elif num_RF_chains == 3:
    X = np.concatenate((X[:, :, 0:2, np.newaxis], X[:, :, 2:4, np.newaxis], X[:, :, 4:6, np.newaxis]), axis=3)
elif num_RF_chains == 2:
    X = np.concatenate((X[:, :, 0:2, np.newaxis], X[:, :, 2:4, np.newaxis]), axis=3)
else:
    X = X[:, :, 0:2, np.newaxis]

train_y = to_categorical(labels, num_classes=None)

print(tensorflow.test.is_gpu_available(cuda_only=True))


class MyHyperModel(kt.HyperModel):
    def build(self, hp, input_shape=None):
        model = Sequential()
        w_size = hp.Int('Window_Size', min_value=128, max_value=710, step=32)
        antenna_size = hp.Int("#antennas", min_value=1, max_value=6, step=1)
        if input_shape:
            model.add(Input(shape=input_shape))
        else:
            model.add(Input(shape=(w_size, 2, antenna_size)))
        # model.add(Input(shape=(710,2,6)))
        ##convolutional settings
        for i in range(hp.Int("CNN_layer_num", 1, 3)):
            if i == 0:
                model.add(Conv2D(filters=hp.Int('ConvL' + str(i) + ' #Filters', min_value=16, max_value=64, step=16),
                                 kernel_size=(
                                 hp.Int('ConvL' + str(i) + 'Stride', min_value=16, max_value=16 * 10, step=16), 2),
                                 activation="relu", padding="same"))
                model.add(Dropout(hp.Choice('DOut' + "Conv" + str(i), values=[0.0, 0.15, 0.3, 0.45, 0.6])))
                model.add(MaxPooling2D(padding='same', pool_size=(
                hp.Int('MaxP' + str(i) + ' Pool size', min_value=20, max_value=80, step=10), 1),
                                       strides=(
                                       hp.Int('MaxP' + str(i) + ' Stride Size', min_value=20, max_value=80, step=10),
                                       1)))
                model.add(Flatten())
            else:
                model.add(Reshape((-1, 1)))
                model.add(Conv1D(filters=hp.Int('ConvL' + str(i) + ' #Filters', min_value=16, max_value=64, step=16),
                                 kernel_size=hp.Int('ConvL' + str(i) + ' Stride', min_value=16, max_value=16 * 10,
                                                    step=16), activation='relu', padding="same"))
                model.add(Dropout(hp.Choice('DOut' + "Conv" + str(i), values=[0.0, 0.15, 0.3, 0.45, 0.6])))
                model.add(MaxPooling1D(padding='same',
                                       pool_size=hp.Int('MaxP' + str(i) + ' Pool size', min_value=20, max_value=80,
                                                        step=10),
                                       strides=hp.Int('MaxP' + str(i) + ' Stride Size', min_value=20, max_value=80,
                                                      step=10)))
                model.add(Flatten())
        ##Fully conntected settings
        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(Dense(units=hp.Int("Dense" + str(i) + " neurons", min_value=5, max_value=50, step=5),
                            activation="relu"))
            model.add(Dropout(hp.Choice('DOut_Dense' + str(i), values=[0.0, 0.15, 0.3, 0.45, 0.6])))
        model.add(Dense(3, activation='softmax'))
        ##optimizer settings
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        opt = optimizers.Adam(
            learning_rate=hp_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
        )
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
        model.summary()
        return model

    def fit(self, hp, model, x, y, validation_data=None, n_antenna=None, **kwargs):
        # processing
        # X_train, X_test, y_train, y_test = train_test_split(x[:,0:hp.get("Window_Size"),:,0:hp.get("#antennas")], y, test_size=0.7, shuffle=True, random_state=seed)
        # w_size=hp.Int('Window_Size', min_value=128, max_value=710, step=32)
        # K.clear_session()
        # cuda.select_device(0)
        # cuda.close()

        if n_antenna:
            filtered_x_train = x[:, 0:hp.get("Window_Size"), :, 0:n_antenna]
            # filtered_x_train = x[:,0:w_size,:,:]
            if validation_data:
                x_val, y_val = validation_data
                filtered_x_val = x_val[:, 0:hp.get("Window_Size"), :, 0:n_antenna]
                # filtered_x_val = x_val[:,0:w_size,:,:]
                validation_data = (filtered_x_val, y_val)
            # X_t, X_v, y_t, y_v = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=seed)
            print(np.shape(filtered_x_train))
            print(np.shape(filtered_x_val))
        else:
            filtered_x_train = x[:, 0:hp.get("Window_Size"), :, 0:hp.get("#antennas")]
            # filtered_x_train = x[:,0:w_size,:,:]
            if validation_data:
                x_val, y_val = validation_data
                filtered_x_val = x_val[:, 0:hp.get("Window_Size"), :, 0:hp.get("#antennas")]
                # filtered_x_val = x_val[:,0:w_size,:,:]
                validation_data = (filtered_x_val, y_val)
            # X_t, X_v, y_t, y_v = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=seed)
            print(np.shape(filtered_x_train))
            print(np.shape(filtered_x_val))
        return model.fit(filtered_x_train, y,
                         shuffle=hp.Boolean("shuffle"),
                         validation_data=validation_data,
                         # batch_size=  hp.Int('Batch_Size', min_value=32, max_value=32*20, step=32),
                         batch_size=hp.Int('Batch_Size', min_value=32, max_value=32 * 10, step=32),
                         **kwargs)

    def predict(self, hp, model, x, n_antenna=None, **kwargs):
        if n_antenna:
            filtered_x = x[:, 0:hp.get("Window_Size"), :, 0:n_antenna]
        else:
            filtered_x = x[:, 0:hp.get("Window_Size"), :, 0:hp.get("#antennas")]
        # filtered_x= x[:,0:hp.get("Window_Size"),:,:]
        return model.predict(filtered_x, **kwargs)

    def evaluate(self, hp, model, x, y, n_antenna=None, **kwargs):
        if n_antenna:
            filtered_x = x[:, 0:hp.get("Window_Size"), :, 0:n_antenna]
        else:
            filtered_x = x[:, 0:hp.get("Window_Size"), :, 0:hp.get("#antennas")]
        # filtered_x= x[:,0:hp.get("Window_Size"),:,:]
        return model.evaluate(filtered_x, y, **kwargs)


tuner = kt.Hyperband(MyHyperModel(),
                     objective='val_accuracy',
                     max_epochs=100,
                     factor=3,
                     overwrite=True,
                     directory=tuner_model_dir,
                     project_name="oc2e_optimalarc_" + string + "_mcs" + str(mcs) + "_rfs" + str(num_RF_chains))

X_train, X_test, y_train, y_test = train_test_split(X, train_y, test_size=0.33, shuffle=True, random_state=seed)

del X
del train_y

stop_early = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='auto', restore_best_weights=True)
print("started")

tuner.search(x=X_train, y=y_train, epochs=1, validation_data=(X_test, y_test), callbacks=[stop_early])

del X_train
del y_train
del X_test
del y_test

# tuner.results_summary(1)
best_hps = tuner.get_best_hyperparameters(1)

with open('/xdisk/krunz/yazdaniabyaneh/802ax_multi_automl/results_rand/oce_opt_10k_' + string + 'best_hyper.pickle',
          'wb') as handle:
    pickle.dump(best_hps[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
