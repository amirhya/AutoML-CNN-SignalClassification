
import tensorflow
from tensorflow.keras.layers import Reshape,SimpleRNN,Dense,Conv1D,LSTM,Dropout, Conv2D, MaxPooling2D, MaxPooling1D, Flatten, Embedding, Bidirectional, Input
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import sys
sys.path.append('./python_codes/')

from windowing import windOwized, LDG

import numpy as np
import pickle

EPOCHS=500 #for OTA



seed=int(sys.argv[2]) # 1...20

normalize=int(sys.argv[3]) #1 or 0

normDict={0:False,1:True}


load_dir="./data/processed/"
tuner_model_dir="./tuner_models/"

print(load_dir)
snr = int(sys.argv[1]) #dB
num_RF_chains=6
string="SNRs_"
snr_set=[snr]
for snr in snr_set:
    string+=str(snr)+"_"

model_num="automml_"+string+"RF"+str(num_RF_chains)+"_seed"+str(seed)
techs = {"w": [], "l": [], "g": []}
tech_labels = {"w": 0, "l": 1, "g": 2}

mcs=64





def multi_class_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.argmax(system, axis=1) == np.argmax(human, axis=1))




print(tensorflow.test.is_gpu_available(cuda_only=True))







class MyHyperModel(kt.HyperModel):
    def build(self, hp, input_shape=None):
        model = Sequential()
        w_size=hp.Int('Window_Size', min_value=128, max_value=710, step=32)
        antenna_size=hp.Int("#antennas", min_value=1, max_value=6, step=1)
        if input_shape:
            model.add(Input(shape=input_shape))
        else:
            model.add(Input(shape=(w_size,2,antenna_size)))
        #model.add(Input(shape=(710,2,6)))
        ##convolutional settings
        for i in range(hp.Int("CNN_layer_num", 1, 3)):
            if i==0:
                model.add(Conv2D(filters=hp.Int('ConvL'+str(i) + ' #Filters', min_value=16, max_value=64, step=16),
                kernel_size=(hp.Int('ConvL'+str(i) + 'Stride', min_value=16, max_value=16*10, step=16),2),activation="relu",padding="same"))
                model.add(Dropout(hp.Choice('DOut'+"Conv"+str(i), values=[0.0, 0.15, 0.3, 0.45, 0.6])))
                model.add(MaxPooling2D(padding='same', pool_size=(hp.Int('MaxP' +str(i)+' Pool size', min_value=5, max_value=80, step=10),1),
                strides=(hp.Int('MaxP'+str(i)+' Stride Size', min_value=5, max_value=60, step=10),1)))
                model.add(Flatten())
            else:
                model.add(Reshape((-1,1)))
                model.add(Conv1D(filters=hp.Int('ConvL'+str(i) + ' #Filters', min_value=16, max_value=32*5, step=16),
                 kernel_size=hp.Int('ConvL'+ str(i)+ ' Stride', min_value=16, max_value=16*10, step=16), activation='relu',padding="same"))
                model.add(Dropout(hp.Choice('DOut'+"Conv"+str(i), values=[0.0, 0.15, 0.3, 0.45, 0.6])))
                model.add(MaxPooling1D(padding='same', pool_size=hp.Int('MaxP' +str(i)+' Pool size', min_value=5, max_value=80, step=10),
                strides=hp.Int('MaxP'+str(i)+' Stride Size', min_value=5, max_value=60, step=10)))
                model.add(Flatten())
        ##Fully conntected settings
        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(Dense(units=hp.Int("Dense" + str(i)+" neurons", min_value=10, max_value=100, step=10), activation="relu"))
            model.add(Dropout(hp.Choice('DOut_Dense'+ str(i), values=[0.0, 0.15, 0.3, 0.45, 0.6])))
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
        model.compile(optimizer=opt, loss="categorical_crossentropy",metrics=['accuracy'])
        #model.summary()
        return model
## need to make sure of proper seperation of train/test/validation data
    def fit(self, hp, model, x, y,validation_data=None, n_antenna=None, **kwargs):
        #processing
        #X_train, X_test, y_train, y_test = train_test_split(x[:,0:hp.get("Window_Size"),:,0:hp.get("#antennas")], y, test_size=0.7, shuffle=True, random_state=seed)
        #w_size=hp.Int('Window_Size', min_value=128, max_value=710, step=32)
        if n_antenna:
            filtered_x_train = x[:,0:hp.get("Window_Size"),:,0:n_antenna]
            #filtered_x_train = x[:,0:w_size,:,:]
            if validation_data:
                x_val, y_val = validation_data
                filtered_x_val = x_val[:,0:hp.get("Window_Size"),:,0:n_antenna]
                #filtered_x_val = x_val[:,0:w_size,:,:]
                validation_data = (filtered_x_val, y_val)
            #X_t, X_v, y_t, y_v = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=seed)
            print(np.shape(filtered_x_train))
            print(np.shape(filtered_x_val))
        else:
            filtered_x_train = x[:,0:hp.get("Window_Size"),:,0:hp.get("#antennas")]
            #filtered_x_train = x[:,0:w_size,:,:]
            if validation_data:
                x_val, y_val = validation_data
                filtered_x_val = x_val[:,0:hp.get("Window_Size"),:,0:hp.get("#antennas")]
                #filtered_x_val = x_val[:,0:w_size,:,:]
                validation_data = (filtered_x_val, y_val)
            #X_t, X_v, y_t, y_v = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=seed)
            print(np.shape(filtered_x_train))
            print(np.shape(filtered_x_val))
        return model.fit(filtered_x_train, y,
            shuffle=hp.Boolean("shuffle"),
            validation_data=validation_data,
            #batch_size=  hp.Int('Batch_Size', min_value=32, max_value=32*20, step=32),
            batch_size=  hp.Int('Batch_Size', min_value=32, max_value=32*10, step=32),
            **kwargs)      
    def predict(self, hp, model, x,  n_antenna=None,  **kwargs):
        if n_antenna:
            filtered_x= x[:,0:hp.get("Window_Size"),:,0:n_antenna]
        else:
            filtered_x= x[:,0:hp.get("Window_Size"),:,0:hp.get("#antennas")]
        #filtered_x= x[:,0:hp.get("Window_Size"),:,:]
        return model.predict(filtered_x, **kwargs)
    def evaluate(self, hp, model, x,y,  n_antenna=None, **kwargs):
        if n_antenna:
            filtered_x= x[:,0:hp.get("Window_Size"),:,0:n_antenna]
        else:
            filtered_x= x[:,0:hp.get("Window_Size"),:,0:hp.get("#antennas")]
        #filtered_x= x[:,0:hp.get("Window_Size"),:,:]
        return model.evaluate(filtered_x,y, **kwargs)







def base_model_train(train_data,validation_data, window, n_antenna):
    X_train,y_train= train_data
    X_train=X_train[:,0:window,:,0:n_antenna]
    x_val, y_val = validation_data
    x_val=x_val[:,0:window,:,0:n_antenna]
    input_shape=np.shape(X_train)[1:]
    base_model = Sequential()
    base_model.add(Input(shape=input_shape))
    base_model.add(Conv2D(filters=64, kernel_size=(101,2),activation="relu"))
    base_model.add(MaxPooling2D(padding='valid', pool_size=(20,1),strides=(20,1)))
    base_model.add(Flatten())
    base_model.add(Dense(10, activation='relu'))
    base_model.add(Dropout(0.3))
    base_model.add(Dense(3, activation='softmax'))
    base_opt = optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
    )
    base_model.compile(
        optimizer=base_opt
        , loss="categorical_crossentropy",metrics=['accuracy'])
    base_clbk = EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='auto', restore_best_weights=True)
    base_model.fit(x=X_train, y=y_train, epochs=EPOCHS, validation_data=(x_val, y_val), callbacks=[base_clbk],batch_size=128)
    return base_model






with open('./results/simulation_'+string+'best_hyper.pickle', 'rb') as handle:
    best_hyper= pickle.load(handle)

size_2=100_000



stop_early =EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='auto', restore_best_weights=True)

results={}


[X, train_y]=LDG(snr, load_dir,512,size_2,num_antennas=6,normalize=normDict[normalize])



X_train, X_test, y_train, y_test = train_test_split(X, train_y, test_size=0.4, shuffle=True, random_state=seed)
del X, train_y
X_t, X_v, y_t, y_v = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=seed)
del X_test, y_test


for n_antenna in [1,2,3,4,5,6]:
    print(str(snr)+"_"+str(n_antenna))

    tuner = kt.Hyperband(MyHyperModel(),
                     objective='val_accuracy',
                     max_epochs=100,
                     factor=3,
                     overwrite=False,
                     directory=tuner_model_dir,
                     project_name="model_simulation"+string+"_mcs64_rfs6")
    tuner.reload()
    m=tuner.get_best_models(num_models=1)[0]
    

    if n_antenna == best_hyper['#antennas']:
        opt_accuracy=multi_class_accuracy(m.predict(X_v[:,0:best_hyper['Window_Size'],:,0:n_antenna]),y_v)


    base_model=base_model_train(train_data=(X_train,y_train),validation_data=(X_v,y_v), window=512, n_antenna=n_antenna)

    base_accuracy = multi_class_accuracy(base_model.predict(X_t[:,0:512,:,0:n_antenna]), y_t)

    if n_antenna == best_hyper['#antennas']:
        results[str(n_antenna)]={"base": base_accuracy, "Opt": opt_accuracy}
    else:
        results[str(n_antenna)]={"base": base_accuracy}




    for snr_2 in [-10,-5,0,5,10,15,20]:
        if snr_2 !=snr:
    
            [X_2,train_y_2]= LDG(snr_2, load_dir,512,size_2,num_antennas=6,normalize=normDict[normalize])



            X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, train_y_2, test_size=0.4, shuffle=True, random_state=seed)
            del X_2, train_y_2
            X_t_2, X_v_2, y_t_2, y_v_2 = train_test_split(X_test_2, y_test_2, test_size=0.5, shuffle=True, random_state=seed)
            del X_train_2,X_test_2,X_v_2, y_train_2, y_test_2,y_v_2

            if n_antenna == best_hyper['#antennas']:
                opt_accuracy = multi_class_accuracy(m.predict(X_t_2[:,0:best_hyper['Window_Size'],:,0:n_antenna]), y_t_2)
            base_accuracy = multi_class_accuracy(base_model.predict(X_t_2[:,0:512,:,0:n_antenna]), y_t_2)
            if n_antenna == best_hyper['#antennas']:
                results[str(n_antenna)][str(snr_2)] ={"base": base_accuracy, "Opt": opt_accuracy}
            else:
                results[str(n_antenna)][str(snr_2)] ={"base": base_accuracy}


if normalize == 1:
    loc= "Normalized"
else:
    loc="notNormalized"

with open("./results/simulation/SEEDs_"+loc+"/OCNN"+string+"simulation_acc_"+loc+"_seed"+str(seed)+".pickle", 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


