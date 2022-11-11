import hickle as hkl
import re
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

num_RF_chains=6
mcs=64
def windOwized(load_data_directory,file_name, shift, length, size,num_antennas):
    max_num_avai_frames=500
    tech_labels = {"w": 0, "l": 1, "g": 2}
    mcs_labels = {"64": 64}
    
    L = re.split('_', file_name)
    y=np.array([tech_labels[L[1]], mcs_labels[L[2]], int(L[3])]) ## technology, mcs, and snr value
    file_path = load_data_directory + file_name
    print("Loading "+file_path)
    waveforms=hkl.load(file_path)
    shapes=np.shape(waveforms)
    effective_length = shapes[1] - (shapes[1] - length) % shift
    Num_IQ = int((effective_length / shift) + 1) - 1
    Num_waveforms = min([size//Num_IQ + 1,max_num_avai_frames])

    X=[]
    print(Num_waveforms)


    for i_wave, wave in enumerate(waveforms[0:Num_waveforms]):
        X=X+[wave[0:effective_length].reshape((shapes[1]//length, length, 2*num_antennas))]
    X=np.array(X)
    shapes=np.shape(X)
    X=X.reshape((shapes[0]*shapes[1],shapes[2],shapes[3]))
    X=X[0:size]
    return [X,y]



def LDG(snr, load_dir,window,  size,num_antennas,normalize=None):
    techs = {"w": [], "l": [], "g": []}
    tech_labels = {"w": 0, "l": 1, "g": 2}
    for tech in techs:
        file_name = "Waveforms_"+tech+"_"+str(mcs)+"_"+str(snr)+"_rf6.hkl"
        techs[tech] = windOwized(load_dir,file_name,  window, window, size,num_RF_chains)[0]
        techs[tech]=np.array(techs[tech])
    min_waveforms=np.min([len(techs[tech]) for tech in techs])
    labels = [] #labels
    features = [] #features
    for tech in techs:
        features.append(techs[tech][0:min_waveforms])
        labels.append(np.ones((size,1))*tech_labels[tech])
    features=np.array(features)
    f_shape=np.shape(features)
    features=features.reshape((f_shape[0]*f_shape[1],f_shape[2],f_shape[3]))
    labels=np.array(labels)
    l_shape=np.shape(labels)
    labels=labels.reshape((l_shape[0]*l_shape[1],l_shape[2]))

    if normalize==True or normalize == None:
        for i, x in enumerate(features):
                scaler = MinMaxScaler()
                scaler.fit(features[i])
                features[i]=scaler.transform(x)
    labels=labels[:,0]
    labels=labels.reshape(-1)
    X=features
    del features
    ##6 RF chains
    if num_antennas==6:
        X=np.concatenate((X[:,:,0:2,np.newaxis], X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis],X[:,:,6:8,np.newaxis], X[:,:,8:10,np.newaxis], X[:,:,10:12,np.newaxis]), axis=3)
    elif num_antennas ==5:
        X=np.concatenate((X[:,:,0:2,np.newaxis], X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis],X[:,:,6:8,np.newaxis], X[:,:,8:10,np.newaxis]), axis=3)

    elif num_antennas==4:
        X=np.concatenate((X[:,:,0:2,np.newaxis], X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis],X[:,:,6:8,np.newaxis]), axis=3)
    elif num_antennas==3:
        X=np.concatenate((X[:,:,0:2,np.newaxis], X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis]), axis=3)
    elif num_antennas==2:
        X=np.concatenate((X[:,:,0:2,np.newaxis], X[:,:,2:4,np.newaxis]), axis=3)
    else:
        X=X[:,:,0:2,np.newaxis] 
    train_y = to_categorical(labels, num_classes=None)
    return [X, train_y]

def ota_lr_data_gen(data,type,num_data,num_antennas, seed, normalize=None):
    np.random.seed(seed) 
    X=[]
    y=[]
    for tech in ['w','l','g']:
        number_of_rows = data[type][tech][0].shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_data, replace=False)
        X+=[data[type][tech][0][random_indices,:]]
        y+=[data[type][tech][1][random_indices]]
    X=np.vstack(X)
    y=np.array(y).reshape(-1)-1
    if normalize==True or normalize == None:
        for i, x in enumerate(X):
            scaler = MinMaxScaler()
            scaler.fit(X[i])
            X[i] = scaler.transform(x)
    y = to_categorical(y, num_classes=None)

    if num_antennas==6:
        X=np.concatenate((X[:,:,0:2,np.newaxis], X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis],X[:,:,6:8,np.newaxis], X[:,:,8:10,np.newaxis], X[:,:,10:12,np.newaxis]), axis=3)
    elif num_antennas ==5:
        X=np.concatenate((X[:,:,0:2,np.newaxis], X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis],X[:,:,6:8,np.newaxis], X[:,:,8:10,np.newaxis]), axis=3)

    elif num_antennas==4:
        X=np.concatenate((X[:,:,0:2,np.newaxis], X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis],X[:,:,6:8,np.newaxis]), axis=3)
    elif num_antennas==3:
        X=np.concatenate((X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis], X[:,:,0:2,np.newaxis]), axis=3)
    elif num_antennas==2:
        X=np.concatenate((X[:,:,2:4,np.newaxis], X[:,:,4:6,np.newaxis]), axis=3)
    else:
        X=X[:,:,2:4,np.newaxis] 
    return [X,y]






