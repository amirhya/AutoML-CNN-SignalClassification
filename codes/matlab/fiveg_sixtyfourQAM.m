function wave=fiveg_sixtyfourQAM(random_seed,wireless_channel)

% Generated by MATLAB(R) 9.10 (R2021a) and 5G Toolbox 2.2 (R2021a).
% Generated on: 30-May-2021 23:11:34

%% Generate Downlink Waveform
% Downlink configuration:
cfgDL = nrDLCarrierConfig('FrequencyRange', 'FR1', ...
    'ChannelBandwidth', 20, ...
    'NCellID', 1, ...
    'NumSubframes', 10);

cfgDL.WindowingPercent = 0;
cfgDL.SampleRate = 20000000;
cfgDL.CarrierFrequency = 0;

% SCS Carriers:
scs1 = nrSCSCarrierConfig( ...
    'SubcarrierSpacing', 15, ...
    'NSizeGrid', 106, ...
    'NStartGrid', 3);
cfgDL.SCSCarriers  = {scs1};

% Bandwidth Parts:
bwp1 = nrWavegenBWPConfig( ...
    'BandwidthPartID', 1, ...
    'Label', 'BWP1', ...
    'SubcarrierSpacing', 15, ...
    'CyclicPrefix', 'normal', ...
    'NSizeBWP', 106, ...
    'NStartBWP', 3);
cfgDL.BandwidthParts  = {bwp1};

% SS Burst:
ssBurst1 = nrWavegenSSBurstConfig( ...
    'BlockPattern', 'Case A', ...
    'TransmittedBlocks', [1 1 1 1], ...
    'Period', 20, ...
    'NCRBSSB', [], ...
    'KSSB', 0, ...
    'DataSource', 'MIB', ...
    'DMRSTypeAPosition', 2, ...
    'CellBarred', false, ...
    'IntraFreqReselection', false, ...
    'PDCCHConfigSIB1', 0, ...
    'SubcarrierSpacingCommon', 15, ...
    'Enable', true, ...
    'Power', 0);
cfgDL.SSBurst  = ssBurst1;

% CORESET:
coreset1 = nrCORESETConfig( ...
    'CORESETID', 0, ...
    'Label', 'CORESET0', ...
    'FrequencyResources', [1 1 1 1 1 1 1 1], ...
    'Duration', 2, ...
    'CCEREGMapping', 'interleaved', ...
    'REGBundleSize', 6, ...
    'InterleaverSize', 2, ...
    'ShiftIndex', 0);
coreset2 = nrCORESETConfig( ...
    'CORESETID', 1, ...
    'Label', 'CORESET1', ...
    'FrequencyResources', [1 1 1 1 1 1 1 1], ...
    'Duration', 2, ...
    'CCEREGMapping', 'interleaved', ...
    'REGBundleSize', 6, ...
    'InterleaverSize', 2, ...
    'ShiftIndex', 0);
cfgDL.CORESET  = {coreset1, coreset2};

% Search spaces:
searchSpaces1 = nrSearchSpaceConfig( ...
    'SearchSpaceID', 1, ...
    'Label', 'SearchSpace1', ...
    'CORESETID', 1, ...
    'SearchSpaceType', 'ue', ...
    'StartSymbolWithinSlot', 0, ...
    'SlotPeriodAndOffset', [1 0], ...
    'Duration', 1, ...
    'NumCandidates', [8 8 4 2 1]);
cfgDL.SearchSpaces  = {searchSpaces1};

% PDCCH:
pdcch1 = nrWavegenPDCCHConfig( ...
    'Enable', true, ...
    'Label', 'PDCCH1', ...
    'Power', 0, ...
    'BandwidthPartID', 1, ...
    'SearchSpaceID', 1, ...
    'AggregationLevel', 8, ...
    'AllocatedCandidate', 1, ...
    'SlotAllocation', 0, ...
    'Period', 1, ...
    'Coding', true, ...
    'DataBlockSize', 20, ...
    'DataSource', 'PN9-ITU', ...
    'RNTI', 1, ...
    'DMRSScramblingID', 2, ...
    'DMRSPower', 0);
cfgDL.PDCCH  = {pdcch1};

% PDSCH DMRS:
pdschDMRS1 = nrPDSCHDMRSConfig( ...
    'DMRSConfigurationType', 1, ...
    'DMRSReferencePoint', 'CRB0', ...
    'NumCDMGroupsWithoutData', 2, ...
    'DMRSTypeAPosition', 2, ...
    'DMRSAdditionalPosition', 0, ...
    'DMRSLength', 1, ...
    'CustomSymbolSet', [], ...
    'DMRSPortSet', [], ...
    'NIDNSCID', [], ...
    'NSCID', 0);

% PDSCH PTRS:
pdschPTRS1 = nrPDSCHPTRSConfig( ...
    'TimeDensity', 1, ...
    'FrequencyDensity', 2, ...
    'REOffset', '00', ...
    'PTRSPortSet', []);

% PDSCH Reserved PRB:
pdschReserved1 = nrPDSCHReservedConfig( ...
    'PRBSet', [], ...
    'SymbolSet', [], ...
    'Period', []);

% PDSCH:
pdsch1 = nrWavegenPDSCHConfig( ...
    'Enable', true, ...
    'Label', 'PDSCH1', ...
    'Power', 0, ...
    'BandwidthPartID', 1, ...
    'Modulation', '64QAM', ...
    'NumLayers', 1, ...
    'MappingType', 'A', ...
    'ReservedCORESET', [], ...
    'SymbolAllocation', [0 14], ...
    'SlotAllocation', [0 1 2 3 4 5 6 7 8 9], ...
    'Period', 10, ...
    'PRBSet', [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105], ...
    'VRBToPRBInterleaving', false, ...
    'VRBBundleSize', 2, ...
    'NID', 1, ...
    'RNTI', 1, ...
    'Coding', true, ...
    'TargetCodeRate', 0.513671875, ...
    'TBScaling', 1, ...
    'XOverhead', 0, ...
    'RVSequence', [0 2 3 1], ...
    'DataSource', {'PN9-ITU',random_seed}, ...
    'DMRSPower', 0, ...
    'EnablePTRS', false, ...
    'PTRSPower', 0);
pdsch1.ReservedPRB{1} = pdschReserved1;
pdsch1.DMRS = pdschDMRS1;
pdsch1.PTRS = pdschPTRS1;
cfgDL.PDSCH  = {pdsch1};

% CSI-RS:
csirs1 = nrWavegenCSIRSConfig( ...
    'Enable', false, ...
    'Label', 'CSIRS1', ...
    'Power', 0, ...
    'BandwidthPartID', 1, ...
    'CSIRSType', 'nzp', ...
    'CSIRSPeriod', 'on', ...
    'RowNumber', 3, ...
    'Density', 'one', ...
    'SymbolLocations', 0, ...
    'SubcarrierLocations', 0, ...
    'NumRB', 52, ...
    'RBOffset', 0, ...
    'NID', 0);
cfgDL.CSIRS  = {csirs1};

% waveform generation:
waveform = nrWaveformGenerator(cfgDL);
wave=wgenerator(waveform,wireless_channel);

end