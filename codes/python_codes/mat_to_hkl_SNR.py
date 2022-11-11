
##This code 


import mat73
import hickle as hkl
import numpy as np
import sys


#Directory of MATLAB generated waveforms
load_directory="./data/orignal/"
#Directory to dump processed data into
unload_directory="./data/processed/"

type = str(sys.argv[1]) # can be "g", "w", or "l": for 5G, Wi-Fi, or LTE

snr = int(sys.argv[2]) # you could set the desired SNR f
mcs="64"

data_dict = mat73.loadmat(load_directory+type+'_'+mcs+'_rf6.mat')

numWaveforms=500

numAntenas=6
Waveforms=[]
for index, wave in enumerate(data_dict['wave']):
    print(str(index+1)+"/"+str(numWaveforms))
    wave=wave[0]
    signal_power = np.mean(np.mean(abs(wave) ** 2, axis=0))
    sigma_2 = 10 ** ((10 * np.log10(signal_power) - snr) / 10)
    cov = [[sigma_2 / 2, 0], [0, sigma_2 / 2]]
    mean = (0, 0)
    m_noise = np.random.multivariate_normal(mean, cov, (len(wave),numAntenas))
    noise = m_noise[:,:,0]+1j*m_noise[:,:,1]
    wave_noisey = wave + noise
    real=np.real(wave_noisey)
    imag=np.imag(wave_noisey)
    wave=[]
    for antenna in range(numAntenas):
        wave.append(real[:,antenna])
        wave.append(imag[:,antenna])
    wave=np.array(wave).T
    Waveforms.append(wave)

#saving constructed waveforms
hkl.dump(np.array(Waveforms), unload_directory+'Waveforms_'+type+"_"+mcs+"_"+str(snr)+'_rf'+numAntenas+'.hkl')


