numWaveforms=500;
wave=cell(numWaveforms,1);

tgax = wlanTGaxChannel;
tgax.SampleRate=20e6;
tgax.DelayProfile = 'Model-D';
tgax.ChannelBandwidth='CBW20';
tgax.CarrierFrequency=2.495e9;
tgax.EnvironmentalSpeed = 0;
tgax.TransmitReceiveDistance = 3.8;
tgax.NumReceiveAntennas = 6;

tgax.LargeScaleFadingEffect = 'Pathloss and shadowing';
tgax.RandomStream = 'mt19937ar with seed';
tgax.Seed = 1;


for i=1:numWaveforms
    
    wave{i}=fiveg_sixtyfourQAM(i,tgax);
    %wave{i}=fiveg_sixtyfourQAM(i,tgax);
    %wave{i}=fiveg_sixtyfourQAM(i,tgax);
    
    i
end    