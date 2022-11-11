function wave=lte_64QAM_r9(random_seed,wireless_channel)
% Generated by MATLAB(R) 9.10 (R2021a) and LTE Toolbox 3.5 (R2021a).
% Generated on: 30-May-2021 23:50:01

%% Generate Downlink RMC Waveform
% Downlink RMC configuration:
cfg = struct('RC', 'R.9', ...
    'DuplexMode', 'FDD', ...
    'NCellID', 0, ...
    'TotSubframes', 10, ...
    'NumCodewords', 1, ...
    'Windowing', 0, ...
    'AntennaPort', 1);

cfg.OCNGPDSCHEnable = 'Off';
cfg.OCNGPDCCHEnable = 'Off';
cfg.PDSCH.TxScheme = 'Port0';
cfg.PDSCH.RNTI = 1;
cfg.PDSCH.Rho = 0;
cfg.PDSCH.RVSeq = [0 1 2 3];
cfg.PDSCH.NHARQProcesses = 8;
cfg.PDSCH.PMISet = 1;
cfg = lteRMCDL(cfg);

% input bit source:
PN_d=randi([0,1],500,9);
pn = comm.PNSequence('Polynomial', 'x9+x5+1', 'InitialConditions', PN_d(random_seed,:));
pn.SamplesPerFrame = sum(cfg.PDSCH.TrBlkSizes(1, :));
in = pn();


% waveform generation:
[waveform, grid, cfg] = lteRMCDLTool(cfg, in);
wave=resample(waveform,125,96); %resampling from 15.36 MHz to 20 MHz
wave=wgenerator(wave,wireless_channel);
end
