function txWaveform = wgenerator(waveStruct,wireless_channel)
%normalize I/Q samples
G=1/sqrt(100*mean(abs(tx.^2)));
                
txWaveform=tgah(G*tx);

end