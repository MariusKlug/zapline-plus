% CLEAN_DATA_WITH_ZAPLINE_PLUS - Removial of frequency artifacts using ZapLine to remove noise from EEG/MEG data. Adds
% automatic detection of the noise frequencies, chunks the data into segments to account for nonstationarities, detects
% the appropriate number of removed components per chunk, based on the individual noise frequency peak and the coponent
% noise scores. If spectral outliers remain above a threshold, the cleaning becomes stricter, if outliers are below a
% threshold, the cleaning becomes laxer. The lower threshold always takes precedence, ensuring a minimal impact on the
% spectrum while cleaning.
% Based on: de Cheveigne, A. (2020) ZapLine: a simple and effective method to remove power line artifacts.
% NeuroImage, 1, 1-13.
%
% Usage:
%
%   >>  [cleanData, zaplineConfig, analyticsResults, plothandles] = clean_data_with_zapline_plus(data,srate,varargin);
%
%
% Required Inputs:
%
%   data                    - MEEG data matrix
%   srate                   - sampling rate in Hz
%
%
% Optional Parameters (these can be entered as <'key',value> pairs OR as a single struct containing relevant parameters!):
%
%   noisefreqs                      - either 'line' or a vector with one or more noise frequencies to be removed. if 
%                                       empty or missing, noise freqs will be detected automatically, if 'line' the
%                                       noise freq will be set to either 50 or 60 Hz, depending on which has higher
%                                       relative power to surroundings.
%   adaptiveNremove                 - bool. if automatic adaptation of number of removed components should be used. (default = 1)
%   fixedNremove                    - fixed number of removed components. if adaptive removal is used, this will be the
%                                       minimum. Will be automatically adapted if "adaptiveSigma" is set to 1. (default = 1)
%   minfreq                         - minimum frequency to be considered as noise when searching for noise freqs automatically.
%                                       (default = 17)
%   maxfreq                         - maximum frequency to be considered as noise when searching for noise freqs automatically.
%                                       (default = 99)
%   detectionWinsize                - window size in Hz for detection of noise peaks (default 6Hz)
%   coarseFreqDetectPowerDiff       - threshold in 10*log10 scale above the average of the spectrum to detect a peak as
%                                       noise freq. (default = 4, meaning a 2.5 x increase of the power over the mean)
%   coarseFreqDetectLowerPowerDiff  - threshold in 10*log10 scale above the average of the spectrum to detect the end of
%                                       a noise freq peak. (default = 1.76, meaning a 1.5 x increase of the power over the mean)
%   searchIndividualNoise           - bool whether or not individual noise peaks should be used instead of the specified
%                                       or found noise on the complete data (default = 1)
%   freqDetectMultFine              - multiplier for the 5% quantile deviation detector of the fine noise frequency
%                                       detection for adaption of sigma thresholds for too strong/weak cleaning (default = 2)
%   detailedFreqBoundsUpper         - frequency boundaries for the fine threshold of too weak cleaning.
%                                       (default = [-0.05 0.05])
%   detailedFreqBoundsLower         - frequency boundaries for the fine threshold of too strong cleaning.
%                                       (default = [-0.4 0.1])
%   maxProportionAboveUpper         - proportion of frequency samples that may be above the upper threshold before
%                                       cleaning is adapted. (default = 0.005)
%   maxProportionBelowLower         - proportion of frequency samples that may be above the lower threshold before
%                                       cleaning is adapted. (default = 0.005)
%   noiseCompDetectSigma            - initial sigma threshold for iterative outlier detection of noise components to be
%                                       removed. Will be automatically adapted if "adaptiveSigma" is set to 1 (default = 3)
%   adaptiveSigma                   - bool. if automatic adaptation of noiseCompDetectSigma should be used. Also adapts
%                                       fixedNremove when cleaning becomes stricter. (default = 1)
%   minsigma                        - minimum when adapting noiseCompDetectSigma. (default = 2.5)
%   maxsigma                        - maximum when adapting noiseCompDetectSigma. (default = 5)
%   chunkLength                     - length of chunks to be cleaned in seconds. if set to 0, automatic chunks will be used.
%                                       (default = 0)
%   segmentLength                   - length of the segments for automatic chunk detection in seconds (default = 1)
%   minChunkLength                  - minimum chunk length of automatic chunk detection in seconds (default = 30)
%   prominenceQuantile              - quantile of the prominence (difference bewtween peak and through) for peak 
%                                       detection of channel covariance for new chunks (default = 0.95)
%   winSizeCompleteSpectrum         - window size in samples of the pwelch function to compute the spectrum of the complete dataset
%                                       for detecting the noise freqs (default = srate*chunkLength)
%   nkeep                           - PCA reduction of components before removal. (default = number of channels)
%   plotResults                     - bool if plot should be created. (default = 1)
%   figBase                         - integer. figure number to be created and plotted in. each iteration of noisefreqs increases
%                                       this number by 1. (default = 100)
%   overwritePlot                   - bool if plot should be overwritten. if not, figbase will be increased by 100 until
%                                       no figure exists (default = 0)
%
%
% Outputs:
%
%   cleanData               - clean EEG data matrix
%   zaplineConfig           - config struct with all used parameters including the found noise frequencies. Can be
%                               re-entered to fully reproduce the previous cleaning
%   analyticsResults        - struct with all relevant analytics results: raw and cleaned log spectra of all channels,
%                               sigma used for detection, proportion of removed power of complete spectrum, noise
%                               frequency, and below noise frequency, ratio of noise power to surroundings before and
%                               after cleaning, proportion of spectral samples above/below the threshold for each
%                               frequency, matrix of number of removed components per noisefreq and chunk, matrix of
%                               artifact component scores per noisefreq and chunk, matrix of individual noise peaks
%                               found per noisefreq and chunk, matrix of whether or not the noise peak exceeded the
%                               threshold, per noisefreq and chunk
%   plothandles             - vector of handles to the created figures
%
%
% Examples:
%
%   EEG.data = clean_data_with_zapline_plus(EEG.data,EEG.srate);
%   [EEG.data, zaplineConfig] = clean_data_with_zapline_plus(EEG.data,EEG.srate);
%   [EEG.data, zaplineConfig, analyticsResults, plothandles] = clean_data_with_zapline_plus(EEG.data,EEG.srate,'adaptiveSigma',0,'chunkLength',200);
%
%
% See also:
%
%   clean_data_with_zapline_plus_eeglab_wrapper, nt_zapline_plus, iterative_outlier_removal, find_next_noisefreq
%
%
% Author: Marius Klug, 2021

function [cleanData, zaplineConfig, analyticsResults, plothandles] = clean_data_with_zapline_plus(data, srate, varargin)

if nargin == 0
    help clean_data_with_zapline_plus
    return
end

disp('Removing frequency artifacts using ZapLine with adaptations for automatic component selection and chunked data.')
disp(' ')
disp(' ')
disp(' ')
disp('---------------- PLEASE CITE ------------------')
disp(' ')
disp('Klug, M., & Kloosterman, N. A. (2022).Zapline-plus: A Zapline extension for automatic and adaptiveremoval of frequency-specific noise artifacts in M/EEG.')
disp('Human Brain Mapping,1–16. https://doi.org/10.1002/hbm.25832')
disp(' ')
disp('-------------------- AND ---------------------')
disp(' ')
disp('de Cheveigne, A. (2020) ZapLine: a simple and effective method to remove power line artifacts.')
disp('NeuroImage, 1, 1-13. https://doi.org/10.1016/j.neuroimage.2019.116356')
disp(' ')
disp('------------------ THANKS! -------------------')
disp(' ')
disp(' ')
disp(' ')

% if the input is a struct, e.g. another zaplineConfig output, create new varargin array with all struct fields to be
% parsed like regular. this should allow perfect reproduction of the cleaning (except figBase)
if nargin == 3 && isstruct(varargin{1})
    zaplineConfig = varargin{1};
    zaplineFields = fieldnames(zaplineConfig);
    varargin = {};
    for i_fieldname = 1:length(zaplineFields)
        varargin{1+(i_fieldname-1)*2} = zaplineFields{i_fieldname};
        varargin{2+(i_fieldname-1)*2} = zaplineConfig.(zaplineFields{i_fieldname});
    end
    
end

% input parsing settings
p = inputParser;
p.CaseSensitive = false;

addRequired(p, 'data', @(x) validateattributes(x,{'numeric'},{'2d'},'clean_EEG_with_zapline','data'))
addRequired(p, 'srate', @(x) validateattributes(x,{'numeric'},{'positive','scalar','integer'},'clean_EEG_with_zapline','srate'))
addOptional(p, 'noisefreqs', [])%, @(x) validateattributes(x,{'numeric','char'},{},'clean_EEG_with_zapline','noisefreqs')) % for some reason i cant make 'char' work here, it leads to errors in the other parameters
addOptional(p, 'fixedNremove', 1, @(x) validateattributes(x,{'numeric'},{'integer','scalar'},'clean_EEG_with_zapline','fixedNremove'));
addOptional(p, 'minfreq', 17, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','minfreq'))
addOptional(p, 'maxfreq', 99, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','maxfreq'))
addOptional(p, 'detectionWinsize', 6, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','detectionWinsize'))
addOptional(p, 'coarseFreqDetectPowerDiff', 4, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','coarseFreqDetectPowerDiff'))
addOptional(p, 'coarseFreqDetectLowerPowerDiff', 1.76091259055681, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','coarseFreqDetectLowerPowerDiff'))
addOptional(p, 'searchIndividualNoise', 1, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','searchIndividualNoise'));
addOptional(p, 'freqDetectMultFine', 2, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','freqDetectMultFine'))
addOptional(p, 'maxProportionAboveUpper', 0.005, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','maxProportionAboveUpper'))
addOptional(p, 'maxProportionBelowLower', 0.005, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','maxProportionBelowLower'))
addOptional(p, 'adaptiveNremove', 1, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','adaptiveNremove'));
addOptional(p, 'noiseCompDetectSigma', 3, @(x) validateattributes(x,{'numeric'},{'scalar','positive'},'clean_EEG_with_zapline','noiseCompDetectSigma'));
addOptional(p, 'adaptiveSigma', 1, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','adaptiveSigma'));
addOptional(p, 'minsigma', 2.5, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','minsigma'))
addOptional(p, 'maxsigma', 5, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','maxsigma'))
addOptional(p, 'chunkLength', 0, @(x) validateattributes(x,{'numeric'},{'scalar','integer'},'clean_EEG_with_zapline','chunkLength'));
addOptional(p, 'winSizeCompleteSpectrum', 300, @(x) validateattributes(x,{'numeric'},{'scalar','integer'},'clean_EEG_with_zapline','winSizeCompleteSpectrum'));
addOptional(p, 'detailedFreqBoundsUpper', [-0.05 0.05], @(x) validateattributes(x,{'numeric'},{'vector'},'clean_EEG_with_zapline','detailedFreqBoundsUpper'))
addOptional(p, 'detailedFreqBoundsLower', [-0.4 0.1], @(x) validateattributes(x,{'numeric'},{'vector'},'clean_EEG_with_zapline','detailedFreqBoundsLower'))
addOptional(p, 'nkeep', 0, @(x) validateattributes(x,{'numeric'},{'scalar','integer','positive'},'clean_EEG_with_zapline','nkeep'));
addOptional(p, 'plotResults', 1, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','plotResults'));
addOptional(p, 'figBase', 100, @(x) validateattributes(x,{'numeric'},{'scalar','integer','positive'},'clean_EEG_with_zapline','figBase'));
addOptional(p, 'figPos', [], @(x) validateattributes(x,{'numeric'},{'vector'},'clean_EEG_with_zapline','figPos'));
addOptional(p, 'overwritePlot', 0, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','plotResults'));
addOptional(p, 'segmentLength', 1, @(x) validateattributes(x,{'numeric'},{'scalar'},'clean_EEG_with_zapline','segmentLength'));
addOptional(p, 'minChunkLength', 30, @(x) validateattributes(x,{'numeric'},{'scalar'},'clean_EEG_with_zapline','minChunkLength'));
addOptional(p, 'prominenceQuantile', 0.95, @(x) validateattributes(x,{'numeric'},{'scalar'},'clean_EEG_with_zapline','prominenceQuantile'));
addOptional(p, 'saveSpectra', 0, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','saveSpectra'));

% parse the input
parse(p,data,srate,varargin{:});

data = p.Results.data;
srate = p.Results.srate;
noisefreqs = p.Results.noisefreqs;
coarseFreqDetectPowerDiff = p.Results.coarseFreqDetectPowerDiff;
coarseFreqDetectLowerPowerDiff = p.Results.coarseFreqDetectLowerPowerDiff;
searchIndividualNoise = p.Results.searchIndividualNoise;
freqDetectMultFine = p.Results.freqDetectMultFine;
maxProportionAboveUpper = p.Results.maxProportionAboveUpper;
maxProportionBelowLower = p.Results.maxProportionBelowLower;
adaptiveNremove = p.Results.adaptiveNremove;
minfreq = p.Results.minfreq;
maxfreq = p.Results.maxfreq;
detectionWinsize = p.Results.detectionWinsize;
adaptiveSigma = p.Results.adaptiveSigma;
minSigma = p.Results.minsigma;
maxSigma = p.Results.maxsigma;
fixedNremove = p.Results.fixedNremove;
chunkLength = p.Results.chunkLength;
winSizeCompleteSpectrum = p.Results.winSizeCompleteSpectrum;
detailedFreqBoundsUpper = p.Results.detailedFreqBoundsUpper;
detailedFreqBoundsLower = p.Results.detailedFreqBoundsLower;
nkeep = p.Results.nkeep;
plotResults = p.Results.plotResults;
figBase = p.Results.figBase;
figPos = p.Results.figPos;
overwritePlot = p.Results.overwritePlot;
segmentLength = p.Results.segmentLength;
minChunkLength = p.Results.minChunkLength;
prominenceQuantile = p.Results.prominenceQuantile;
saveSpectra = p.Results.saveSpectra;

% finalize inputs

if srate > 500
    warning(sprintf(['\n--------------------------------------- WARNING -----------------------------------------------',...
        '\n\nIt is recommended to downsample the data to around 250Hz to 500Hz before applying Zapline-plus!\n\n',...
        '                      Current srate is ' num2str(srate) '. Results may be suboptimal!\n\n',...
        '--------------------------------------- WARNING -----------------------------------------------']))
end



while ~overwritePlot && ishandle(figBase+1)
    figBase = figBase+100;
end

transposeData = size(data,2)>size(data,1);
if transposeData
    data = data';
end

% we want at least 8 segment fro proper usage of pwelch
if winSizeCompleteSpectrum*srate > size(data,1)/8
    
    winSizeCompleteSpectrum = floor(length(data)/srate/8);
    warning('Data set is short. Adjusted window size for whole data set spectrum calculation to be 1/8 of the length!')
    
end

if nkeep == 0
    % our tests show actually better cleaning performance when no PCA reduction is used!
    
    %     nkeep = min(round(20+size(data,2)/4),size(data,2));
    %     disp(['Reducing the number of components to ' num2str(nkeep) ', set the ''nkeep'' flag to decide otherwise.'])
    nkeep = size(data,2);
end

% create config struct for zapline, also store any additional input for the record
zaplineConfig.noisefreqs = p.Results.noisefreqs;
zaplineConfig.coarseFreqDetectPowerDiff = p.Results.coarseFreqDetectPowerDiff;
zaplineConfig.coarseFreqDetectLowerPowerDiff = p.Results.coarseFreqDetectLowerPowerDiff;
zaplineConfig.searchIndividualNoise = p.Results.searchIndividualNoise;
zaplineConfig.freqDetectMultFine = p.Results.freqDetectMultFine;
zaplineConfig.maxProportionAboveUpper = p.Results.maxProportionAboveUpper;
zaplineConfig.maxProportionBelowLower = p.Results.maxProportionBelowLower;
zaplineConfig.minfreq = p.Results.minfreq;
zaplineConfig.maxfreq = p.Results.maxfreq;
zaplineConfig.detectionWinsize = p.Results.detectionWinsize;
zaplineConfig.adaptiveNremove = p.Results.adaptiveNremove;
zaplineConfig.adaptiveSigma = p.Results.adaptiveSigma;
zaplineConfig.minSigma = p.Results.minsigma;
zaplineConfig.maxSigma = p.Results.maxsigma;
zaplineConfig.fixedNremove = p.Results.fixedNremove;
zaplineConfig.noiseCompDetectSigma = p.Results.noiseCompDetectSigma;
zaplineConfig.chunkLength = chunkLength;
zaplineConfig.winSizeCompleteSpectrum = winSizeCompleteSpectrum;
zaplineConfig.detailedFreqBoundsUpper = p.Results.detailedFreqBoundsUpper;
zaplineConfig.detailedFreqBoundsLower = p.Results.detailedFreqBoundsLower;
zaplineConfig.nkeep = nkeep;
zaplineConfig.segmentLength = segmentLength;
zaplineConfig.minChunkLength = minChunkLength;
zaplineConfig.prominenceQuantile = prominenceQuantile;


% initialize results in case no noise frequenc is found
[pxx_clean_log resSigmaFinal resProportionRemoved resProportionRemovedNoise resProportionRemovedBelowNoise resProportionBelowLower...
    resProportionAboveUpper resRatioNoiseRaw resRatioNoiseClean resNremoveFinal resScores resNoisePeaks resFoundNoise] = deal([]);

%% Clean each frequency one after another

% find flat channels and store, remove from dataset to work on

diffdata = diff(data);
flat_channels_idx = find(all(diffdata==0));
if ~isempty(flat_channels_idx)
    warning(['Flat channels detected (will be ignored and added back in after Zapline-plus processing): ' num2str(flat_channels_idx)])
    flat_channels_data = data(:,flat_channels_idx);
    
    data(:,flat_channels_idx) = [];
end

cleanData = data;

disp('Computing initial spectrum...')
% compute spectrum with frequency resolution of winSizeCompleteSpectrum
[pxx_raw_log,f]=pwelch(data,hanning(winSizeCompleteSpectrum*srate),[],[],srate);
% log transform
pxx_raw_log = 10*log10(pxx_raw_log);

% store initial raw spectrum
if saveSpectra
    analyticsResults.rawSpectrumLog = pxx_raw_log;
    analyticsResults.frequencies = f;
end

% search for line only
lineonly = 0;
if strcmp(noisefreqs,'line')
    
    lineonly = 1;
    % relative 50 Hz power

    % BF_wider_linefrequency_search_range (Suddha Sourav):
    % Previously line frequency was searched in the following ranges:
    % (49.9 Hz < f < 50.1 Hz) OR (59.9 Hz < f < 60.1 Hz). The frequency range
    % of 0.2 Hz is generally sufficient but still might miss line frequencies
    % in some inopportune intervals, see:
    % Schäfer et al. (2018). Non-Gaussian power grid
    % frequency fluctuations characterized by Lévy-stable laws and superstatis-
    % tics. Nature Energy, 3(2), 119-126. doi:
    % https://doi.org/10.1038/s41560-017-0058-z
    %
    % This problem is more serious for EEG/MEG research in lower/middle-income
    % countries, where the range might be wider, see:
    % Gautam et al. (2020). Analyses of Indian Power System Frequency. In 2020
    % IEEE POWERCON (pp. 1-6). IEEE. doi:
    % https://doi.org/10.1109/POWERCON48463.2020.9230532
    %
    % Suggestion: increase the range to 2 Hz, i.e.
    % (49 Hz < f < 51 Hz) OR (59 Hz < f < 60 Hz)

    idx = (f > 49 & f < 50) | (f > 59 & f < 60);
    
    % BF_noise_frequency_candidate_search (Suddha Sourav):
    % Index in 2D, take all channels (i.e. columns) into account by
    % getting a chunk of the spectra over all electrodes at the fre-
    % quencies of interest.
    spectraChunk_allChans = pxx_raw_log(idx,:);
    
    % Get the global maximum across all channels, calculated on the 
    % flattened spectral data chunk.
    [maxVal, n] = max(spectraChunk_allChans(:));
    
    % Find out which row and column (frequency index in the spectral data
    % chunk, and channel number) the max value was in
    [fIdx_max, chanIdx_max] = ind2sub(size(spectraChunk_allChans),n);
    
    % Find out the frequency: first, relate the spectral data chunk's
    % indices to the actual frequency indices, then index based on this
    % vector
    f_spectraChunk_allChans = f(find(idx));
    noisefreqs_candidate = f_spectraChunk_allChans(fIdx_max);
    
    % P.S. for multiple maximum values, the method anove will always return
    % the first maximum value, thus one less potential bug
    
    fprintf('"noisefreqs" parameter was set to ''line'', found line noise candidate at %g Hz!\n',noisefreqs_candidate);
    
    noisefreqs = [];
    minfreq = noisefreqs_candidate-detectionWinsize/2;
    maxfreq = noisefreqs_candidate+detectionWinsize/2;
    
end

automaticFreqDetection = isempty(noisefreqs);
if automaticFreqDetection
    disp(['Searching for first noise frequency between ' num2str(minfreq) ' and ' num2str(maxfreq) 'Hz...'])
    verbose = 0;
    [noisefreqs,~,~,thresh]=find_next_noisefreq(pxx_raw_log,f,minfreq,coarseFreqDetectPowerDiff,detectionWinsize,maxfreq,...
        coarseFreqDetectLowerPowerDiff,verbose);
end

i_noisefreq = 1;
while i_noisefreq <= length(noisefreqs)
    
    noisefreq = noisefreqs(i_noisefreq);
    
    thisFixedNremove = fixedNremove;
    
    fprintf('Removing noise at %gHz... \n',noisefreq);
    
    figThis = figBase+i_noisefreq;
    
    cleaningDone = 0;
    cleaningTooStongOnce = 0;
    thisZaplineConfig = zaplineConfig;
    
    if chunkLength ~= 0
        fprintf('Using fixed chunk length of %.0f seconds!\n', chunkLength)
        chunkIndices = 1;
        while chunkIndices(end) < length(data)-chunkLength*2*srate
            chunkIndices(end+1) = chunkIndices(end)+chunkLength*srate;
        end
        chunkIndices(end+1) = length(data)+1;
    else
        disp('Using adaptive chunk length!')
        %% find chunk indices
        data_narrowfilt = bandpass(data,[noisefreq-detectionWinsize/2 noisefreq+detectionWinsize/2],srate);
        
        nSegments = max(floor(size(data_narrowfilt,1)/srate/segmentLength),1);
        
        covarianceMatrices = zeros(size(data_narrowfilt,2),size(data_narrowfilt,2),nSegments);
        
        %% compute covmatrices
        for iSegment = 1:nSegments
            
            if iSegment ~= nSegments
                segmentIndices = 1+segmentLength*srate*(iSegment-1):segmentLength*srate*(iSegment);
            else
                segmentIndices = 1+segmentLength*srate*(iSegment-1):size(data_narrowfilt,1);
            end
            
            segment = data_narrowfilt(segmentIndices,:);
            
            covarianceMatrices(:,:,iSegment) = cov(segment);
            
        end
        
        %% find distances
        distances = zeros(nSegments-1,1);
        
        for iSegment = 2:nSegments
            
            distances(iSegment-1) = sum(pdist(covarianceMatrices(:,:,iSegment)-covarianceMatrices(:,:,iSegment-1)))/2;
            
        end
        
        %% find peaks
        [pks,locs,widths,proms] = findpeaks(distances);
        [pks,locs] = findpeaks(distances,'MinPeakProminence',quantile(proms,prominenceQuantile),'MinPeakDistance',minChunkLength);
        
        %% plot
        
%         figure('color','w');
%         plot(distances)
%         
%         hold on
%         
%         l = plot(locs,pks,'ko')
%         title('noise narrowband covariance matrix distances')
%         legend(l,'chunk segmentations')
%         xlabel('time [seconds]')
        
        %% create final chunk indices
        
        chunkIndices = ones(length(pks)+2,1);
        
        chunkIndices(2:end-1) = locs*segmentLength*srate;
        
        chunkIndices(end) = length(data)+1;
        
        if chunkIndices(2) - chunkIndices(1) < minChunkLength*srate
            % make sure the last chunk is also min length
            chunkIndices(2) = [];
        end
        
        if chunkIndices(end) - chunkIndices(end-1) < minChunkLength*srate
            % make sure the last chunk is also min length
            chunkIndices(end-1) = [];
        end
    end
    
    nChunks = length(chunkIndices)-1;
    
    fprintf('%.0f chunks will be created.\n', nChunks)
    
    while ~cleaningDone
        
        % result data matrix
        cleanData = NaN(size(data));
        
        % last chunk must be larger than the others, to ensure fft works, at least 1 chunk must be used
        %         nChunks = max(floor(size(data,1)/srate/chunkLength),1);
        scores = NaN(nChunks,nkeep);
        NremoveFinal = NaN(nChunks,1);
        noisePeaks = NaN(nChunks,1);
        foundNoise = zeros(nChunks,1);
        
        for iChunk = 1:nChunks
            
            this_zaplineConfig_chunk = thisZaplineConfig;
            
            if mod(iChunk,round(nChunks/10))==0
                disp(['Chunk ' num2str(iChunk) ' of ' num2str(nChunks)])
            end
            
            %             if iChunk ~= nChunks
            %                 chunkIndices = 1+chunkLength*srate*(iChunk-1):chunkLength*srate*(iChunk);
            %             else
            %                 chunkIndices = 1+chunkLength*srate*(iChunk-1):size(data,1);
            %             end
            
            chunk = data(chunkIndices(iChunk):chunkIndices(iChunk+1)-1,:);
            
            % find flat channels and store, remove from dataset to work on

            diffchunk = diff(chunk);
            flat_channels_idx_chunk = find(all(diffchunk==0));
            if ~isempty(flat_channels_idx_chunk)
                warning(['Chunk ' num2str(iChunk) ': Flat channels detected in chunk (will be ignored and added back in after Zapline-plus processing): ' num2str(flat_channels_idx_chunk)])
                flat_channels_data_chunk = chunk(:,flat_channels_idx_chunk);

                chunk(:,flat_channels_idx_chunk) = [];
            end
            
            if searchIndividualNoise
                % compute spectrum with maximal frequency resolution per chunk to detect individual peaks
                [pxx_chunk,f]=pwelch(chunk,hanning(length(chunk)),[],[],srate);
                pxx_chunk = 10*log10(pxx_chunk);
                
                thisFreqidx = f>noisefreq-(detectionWinsize/2) & f<noisefreq+(detectionWinsize/2);
                this_freq_idx_detailed = f>noisefreq+detailedFreqBoundsUpper(1) & f<noisefreq+detailedFreqBoundsUpper(2);
                this_freqs_detailed = f(this_freq_idx_detailed);
                
                % mean per channels
                thisFineData = mean(pxx_chunk(thisFreqidx,:),2);
                % don't look at middle third, but check left and right around target frequency
                third = round(length(thisFineData)/3);
                centerThisData = mean(thisFineData([1:third third*2:end]));
                
                % use lower quantile as indicator of variability, because upper quantiles may be misleading around the noise
                % frequencies
                meanLowerQuantileThisData = mean([quantile(thisFineData(1:third),0.05) quantile(thisFineData(third*2:end),0.05)]);
                detailedNoiseThresh = centerThisData + freqDetectMultFine * (centerThisData - meanLowerQuantileThisData);
                
                % find peak frequency that is above the threshold
                maxFinePower = max(mean(pxx_chunk(this_freq_idx_detailed,:),2));
                noisePeaks(iChunk) = this_freqs_detailed(mean(pxx_chunk(this_freq_idx_detailed,:),2) == maxFinePower);
                
                if maxFinePower > detailedNoiseThresh
                    % use adaptive cleaning
                    foundNoise(iChunk) = 1;
                    
                else
                    % no noise was found in chunk -> clean with fixed threshold to be sure (it might be a miss of the
                    % detector), but use overall noisefreq
                    noisePeaks(iChunk) = noisefreq;
                    
                    this_zaplineConfig_chunk.adaptiveNremove = 0;
                    
                end
                
            else
                noisePeaks(iChunk) = noisefreq;
            end
            
            %             figure; plot(f,mean(pxx_chunk,2));
            %             xlim([f(find(this_freq_idx,1,'first')) f(find(this_freq_idx,1,'last'))])
            %             hold on
            %             plot([f(find(this_freq_idx_detailed,1,'first')) f(find(this_freq_idx_detailed,1,'last'))],...
            %                 [detailedNoiseThresh detailedNoiseThresh],'r')
            %             plot(xlim,[center_thisdata center_thisdata])
            %             plot(xlim,[mean_lower_quantile_thisdata mean_lower_quantile_thisdata])
            %             title(['chunk ' num2str(iChunk) ', ' num2str(noisePeaks(iChunk))])
            
            % needs to be normalized for zapline
            f_noise = noisePeaks(iChunk)/srate;
            
            % apply Zapline
            [cleanData_chunk,~,NremoveFinal(iChunk),thisScores] =...
                nt_zapline_plus(chunk,f_noise,thisFixedNremove,this_zaplineConfig_chunk,0);
            
            scores(iChunk,1:length(thisScores)) = thisScores;
            
            %             [pxx_chunk,f]=pwelch(cleanData(chunkIndices,:),hanning(length(chunk)),[],[],srate);
            %             pxx_chunk = 10*log10(pxx_chunk);
            %             figure; plot(f,mean(pxx_chunk,2));
            %             xlim([f(find(this_freq_idx,1,'first')) f(find(this_freq_idx,1,'last'))])
            %             title(['chunk ' num2str(iChunk) ', ' num2str(noisePeaks(iChunk)) ', ' num2str(NremoveFinal(iChunk)) ' removed'])
            
            % add flat channels back in
            if ~isempty(flat_channels_idx_chunk)
%                 warning(['Chunk ' num2str(iChunk) ': Detected flat channels in chunk were ignored and are added back in after Zapline plus processing: ' num2str(flat_channels_idx_chunk)])

                fullCleanData_chunk = [];

                i_last = 1;
                i_last_clean = 1;

                for i_flatchan = 1:length(flat_channels_idx_chunk)

                    flatchan = flat_channels_idx_chunk(i_flatchan);
                    fullCleanData_chunk(:,i_last:flatchan-1) = cleanData_chunk(:,i_last_clean:flatchan-i_flatchan);
                    fullCleanData_chunk(:,flatchan) = flat_channels_data_chunk(:,i_flatchan);

                    i_last = flatchan+1;
                    i_last_clean = flatchan-i_flatchan+1;

                end

                fullCleanData_chunk(:,i_last:size(cleanData_chunk,2)+length(flat_channels_idx_chunk)) = cleanData_chunk(:,i_last_clean:end);

                cleanData_chunk = fullCleanData_chunk;

            end
            
            cleanData(chunkIndices(iChunk):chunkIndices(iChunk+1)-1,:) = cleanData_chunk;
            
        end
        disp('Done. Computing spectra...')
        
        % compute spectra
        [pxx_raw]=pwelch(data,hanning(winSizeCompleteSpectrum*srate),[],[],srate);
        pxx_raw_log = 10*log10(pxx_raw);
        [pxx_clean,f]=pwelch(cleanData,hanning(winSizeCompleteSpectrum*srate),[],[],srate);
        pxx_clean_log = 10*log10(pxx_clean);
        [pxx_removed]=pwelch(data-cleanData,hanning(winSizeCompleteSpectrum*srate),[],[],srate);
        pxx_removed_log = 10*log10(pxx_removed);
        
        % compute analytics
        
        % in original space
        proportionRemoved = (mean(pxx_raw(:)) - mean(pxx_clean(:)))/ mean(pxx_raw(:));
        % in log space -> makes more sense to be consistent with visuals, and we argue that the geometric mean is a
        % better measure anyways
        proportionRemoved = 1-10^((mean(pxx_clean_log(:)) - mean(pxx_raw_log(:)))/10);
        disp(['proportion of removed power: ' num2str(proportionRemoved)]);
        
        this_freq_idx_belownoise = f>=max(noisefreq-11,0) & f<=noisefreq-1;
        proportionRemovedBelowNoise = (mean(pxx_raw(this_freq_idx_belownoise,:),'all') - mean(pxx_clean(this_freq_idx_belownoise,:),'all')) /...
            mean(pxx_raw(this_freq_idx_belownoise,:),'all');
        proportionRemovedBelowNoise = 1-10^((mean(pxx_clean_log(this_freq_idx_belownoise,:),'all') - mean(pxx_raw_log(this_freq_idx_belownoise,:),'all'))/10);
        
        (mean(pxx_raw_log(this_freq_idx_belownoise,:),'all') - mean(pxx_clean(this_freq_idx_belownoise,:),'all')) /...
            mean(pxx_raw(this_freq_idx_belownoise,:),'all');
        disp(['proportion of removed power below noise frequency: ' num2str(proportionRemovedBelowNoise)]);
        
        this_freq_idx_noise = f>noisefreq+detailedFreqBoundsUpper(1) & f<noisefreq+detailedFreqBoundsUpper(2);
        proportionRemovedNoise = (mean(pxx_raw(this_freq_idx_noise,:),'all') - mean(pxx_clean(this_freq_idx_noise,:),'all')) /...
            mean(pxx_raw(this_freq_idx_noise,:),'all');
        proportionRemovedNoise = 1-10^((mean(pxx_clean_log(this_freq_idx_noise,:),'all') - mean(pxx_raw_log(this_freq_idx_noise,:),'all'))/10);
        disp(['proportion of removed power at noise frequency: ' num2str(proportionRemovedNoise)]);
        
        this_freq_idx_noise_surrounding = (f>noisefreq-(detectionWinsize/2) & f<noisefreq-(detectionWinsize/6)) |...
            (f>noisefreq+(detectionWinsize/6) & f<noisefreq+(detectionWinsize/2));
        
        ratioNoiseRaw = 10^((mean(mean(pxx_raw_log(this_freq_idx_noise,:),2)) - mean(pxx_raw_log(this_freq_idx_noise_surrounding,:),'all'))/10);
        ratioNoiseClean = 10^((mean(mean(pxx_clean_log(this_freq_idx_noise,:),2)) - mean(pxx_clean_log(this_freq_idx_noise_surrounding,:),'all'))/10);
        
        disp(['ratio of noise power to surroundings power before cleaning: ' num2str(ratioNoiseRaw)]);
        disp(['ratio of noise power to surroundings power after cleaning: ' num2str(ratioNoiseClean)]);
        
        
        
        % check if cleaning was too weak or too strong
        
        % determine center power by checking lower and upper third around noise freq, then check detailed lower and
        % upper threhsold. search area for weak is around the noisefreq, for strong its larger and a little below the
        % noisefreq because zapline makes a dent there
        thisFreqidx = f>noisefreq-(detectionWinsize/2) & f<noisefreq+(detectionWinsize/2);
        thisFreqidxUppercheck = f>noisefreq+detailedFreqBoundsUpper(1) & f<noisefreq+detailedFreqBoundsUpper(2);
        thisFreqidxLowercheck = f>noisefreq+detailedFreqBoundsLower(1) & f<noisefreq+detailedFreqBoundsLower(2);
        
        thisFineData = mean(pxx_clean_log(thisFreqidx,:),2);
        third = round(length(thisFineData)/3);
        centerThisData = mean(thisFineData([1:third third*2:end]));
        
        % measure of variation in this case is only lower quantile because upper quantile can be driven by spectral outliers
        meanLowerQuantileThisData = mean([quantile(thisFineData(1:third),0.05) quantile(thisFineData(third*2:end),0.05)]);
        
        remainingNoiseThreshUpper = centerThisData + freqDetectMultFine * (centerThisData - meanLowerQuantileThisData);
        remainingNoiseThreshLower = centerThisData - freqDetectMultFine * (centerThisData - meanLowerQuantileThisData);
        
        % if x% of the samples in the search area are below or above the thresh it's too strong or weak
        proportionAboveUpper = sum(mean(pxx_clean_log(thisFreqidxUppercheck,:),2) > remainingNoiseThreshUpper) / sum(thisFreqidxUppercheck);
        cleaningTooWeak =  proportionAboveUpper > maxProportionAboveUpper;
        
        proportionBelowLower = sum(mean(pxx_clean_log(thisFreqidxLowercheck,:),2) < remainingNoiseThreshLower) / sum(thisFreqidxLowercheck);
        cleaningTooStong = proportionBelowLower > maxProportionBelowLower;
        
        disp([num2str(round(proportionAboveUpper*100,2)) '% of frequency samples above thresh in the range of '...
            num2str(detailedFreqBoundsUpper(1)) ' to ' num2str(detailedFreqBoundsUpper(2)) 'Hz around noisefreq (threshold is '...
            num2str(maxProportionAboveUpper*100) '%).'])
        disp([num2str(round(proportionBelowLower*100,2)) '% of frequency samples below thresh in the range of '...
            num2str(detailedFreqBoundsLower(1)) ' to ' num2str(detailedFreqBoundsLower(2)) 'Hz around noisefreq (threshold is '...
            num2str(maxProportionBelowLower*100) '%).'])
        
        if plotResults
            
            %%
            chunkIndicesPlot = chunkIndices/srate/60; % for plotting convert to minutes
            chunkIndicesPlotIndividual = [];
            for i_chunk = 1:length(chunkIndicesPlot)-1
                chunkIndicesPlotIndividual(i_chunk) = mean([chunkIndicesPlot(i_chunk),chunkIndicesPlot(i_chunk+1)]);
            end
            
            
            red = [230 100 50]/256;
            green = [0 97 100]/256;
            grey = [0.2 0.2 0.2];
            
            this_freq_idx_plot = f>=noisefreq-1.1 & f<=noisefreq+1.1;
            plothandles(i_noisefreq) = figure(figThis);clf; 
            if ~isempty(figPos)
                set(gcf,'color','w','Position',figPos) % e.g. figpos = [0 0 1500 850] 
            else
                set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized','outerposition',[0.2 0.2 0.7 0.7])
            end
            set(gcf,'name',[num2str(noisefreq,'%4.2f') 'Hz'])
            
            % plot original power
            subplot(3,30,[1:5]);
            
            plot(f(this_freq_idx_plot),mean(pxx_raw_log(this_freq_idx_plot,:),2),'color',grey)
            xlim([f(find(this_freq_idx_plot,1,'first'))-0.01 f(find(this_freq_idx_plot,1,'last'))])
            
            
            ylim([remainingNoiseThreshLower-0.25*(remainingNoiseThreshUpper-remainingNoiseThreshLower)
                min(mean(pxx_raw_log(this_freq_idx_plot,:),2))+coarseFreqDetectPowerDiff*2])
            box off
            
            hold on
            if automaticFreqDetection && ~lineonly
                plot(xlim,[thresh thresh],'color',red)
                title({'detected frequency:', [num2str(noisefreq,'%4.2f') 'Hz']})
            elseif automaticFreqDetection && lineonly
                plot(xlim,[thresh thresh],'color',red)
                title({'detected line frequency:', [num2str(noisefreq,'%4.2f') 'Hz']})
            else
                title({'predefined frequency:', [num2str(noisefreq,'%4.2f') 'Hz']})
            end
            xlabel('frequency [Hz]')
            ylabel('Power [10*log10 \muV^2/Hz]')
            set(gca,'fontsize',12)
            
            
            % plot nremoved
            pos = 8:17;
            subplot(24,60,[pos pos+30]*2-1);cla
            hold on
            
            for i_chunk = 1:length(chunkIndicesPlot)-1

                if ~searchIndividualNoise || foundNoise(i_chunk)
                    fill([chunkIndicesPlot(i_chunk) chunkIndicesPlot(i_chunk) chunkIndicesPlot(i_chunk+1) chunkIndicesPlot(i_chunk+1)],...
                        [0 NremoveFinal(i_chunk) NremoveFinal(i_chunk) 0],grey,'facealpha',0.5)
                else
                    nonoisehandle = fill([chunkIndicesPlot(i_chunk) chunkIndicesPlot(i_chunk) chunkIndicesPlot(i_chunk+1) chunkIndicesPlot(i_chunk+1)],...
                        [0 NremoveFinal(i_chunk) NremoveFinal(i_chunk) 0],green,'facealpha',0.5);
                end
            end
            
            xlim([chunkIndicesPlot(1) chunkIndicesPlot(end)])
            ylim([0 max(NremoveFinal)+1])
            title({['# removed comps in ' num2str(nChunks)...
                ' chunks, \mu = ' num2str(round(mean(NremoveFinal),2))]})
            set(gca,'fontsize',12)
%             if searchIndividualNoise
%                 foundNoisePlot = foundNoise;
%                 foundNoisePlot(foundNoisePlot==1) = NaN;
%                 foundNoisePlot(~isnan(foundNoisePlot)) = NremoveFinal(~isnan(foundNoisePlot));
%                 plot(chunkIndicesPlotIndividual,foundNoisePlot,'o','color',green);
%             end
            box off
            
            % plot noisepeaks
            subplot(24*2,60,[pos+30*9 pos+30*10 pos+30*11 pos+30*12]*2-1);cla % lol dont judge me it works
            hold on
            
            for i_chunk = 1:length(chunkIndicesPlot)-2
                plot([chunkIndicesPlot(i_chunk+1) chunkIndicesPlot(i_chunk+1)],[0 1000],'color',grey*3)
%                     fill([chunkIndicesPlot(i_chunk) chunkIndicesPlot(i_chunk) chunkIndicesPlot(i_chunk+1) chunkIndicesPlot(i_chunk+1)],...
%                         [noisePeaks(i_chunk) noisePeaks(i_chunk) noisePeaks(i_chunk) noisePeaks(i_chunk)],grey)
            end
            
            plot(chunkIndicesPlotIndividual,[noisePeaks],'-o','color',grey,'markerfacecolor',grey,'markersize',3)
            xlim([chunkIndicesPlot(1) chunkIndicesPlot(end)])
            maxdiff = max([(max(noisePeaks))-noisefreq noisefreq-(min(noisePeaks))]);
            if maxdiff == 0
                maxdiff = 0.01;
            end
            ylim([noisefreq-maxdiff*1.5 noisefreq+maxdiff*1.5])
            xlabel('time [minutes]')
            title({['individual peak frequencies [Hz]']})
            if searchIndividualNoise
                foundNoisePlot = foundNoise;
                foundNoisePlot(foundNoisePlot==1) = NaN;
                foundNoisePlot(~isnan(foundNoisePlot)) = noisePeaks(~isnan(foundNoisePlot));
                plot(chunkIndicesPlotIndividual,foundNoisePlot,'s','color',green,'markerfacecolor',green,'markersize',8);
                
                if exist('nonoisehandle','var')
                    legend(nonoisehandle,{'no clear noise peak found'},'edgecolor',[0.8 0.8 0.8],'position',...
                        [0.368923614106865 0.805246914159736 0.127083330337579 0.023148147568658]);
                end
            end
            box off
            set(gca,'fontsize',12)
            
            % plot scores
            subplot(3,30,[19:23]);
            
            plot(nanmean(scores,1),'color',grey)
            hold on
            meanremovedhandle = plot([mean(NremoveFinal)+1 mean(NremoveFinal)+1],ylim,'color',red);
            xlim([0.7 round(size(scores,2)/3)])
            if adaptiveNremove
                title({'mean artifact scores [a.u.]', ['\sigma for detection = ' num2str(thisZaplineConfig.noiseCompDetectSigma)]})
            else
                title({'mean artifact scores [a.u.]'})
            end
            xlabel('component')
            set(gca,'fontsize',12)
            box off
            legend(meanremovedhandle, 'mean removed','edgecolor',[0.8 0.8 0.8])
            
            % plot new power
            subplot(3,30,[26:30]);
            
            hold on
            plot(f(this_freq_idx_plot),mean(pxx_clean_log(this_freq_idx_plot,:),2),'color', green)
            
            xlim([f(find(this_freq_idx_plot,1,'first'))-0.01 f(find(this_freq_idx_plot,1,'last'))])
            
            
            try
                % this wont work if the frequency resolution is too low
                l1 = plot([f(find(thisFreqidxUppercheck,1,'first')) f(find(thisFreqidxUppercheck,1,'last'))],...
                    [remainingNoiseThreshUpper remainingNoiseThreshUpper],'color',grey);
                l2 = plot([f(find(thisFreqidxLowercheck,1,'first')) f(find(thisFreqidxLowercheck,1,'last'))],...
                    [remainingNoiseThreshLower remainingNoiseThreshLower],'color',red);
                legend([l1 l2], {[num2str(round(proportionAboveUpper*100,2)) '% above']
                    [num2str(round(proportionBelowLower*100,2)) '% below']},...
                    'location','north','edgecolor',[0.8 0.8 0.8])
            end
            ylim([remainingNoiseThreshLower-0.25*(remainingNoiseThreshUpper-remainingNoiseThreshLower)
                min(mean(pxx_raw_log(this_freq_idx_plot,:),2))+coarseFreqDetectPowerDiff*2])
           
            
            xlabel('frequency [Hz]')
            ylabel('Power [10*log10 \muV^2/Hz]')
            title('cleaned spectrum')
            set(gca,'fontsize',12)
            box off
            
            
            % plot starting spectrum
            
            pos = [11:14 21:24];
            ax1 = subplot(60,10,[pos+60*4 pos+60*5 pos+60*6 pos+60*7 pos+60*8 pos+60*9]);
            hold on
            cla
            %             singlehandles = plot(f,pxx_raw_log,'color',[0.8 0.8 0.8]);
            meanhandles = plot(f,mean(pxx_raw_log,2),'color',grey,'linewidth',1.5);
            set(gca,'ygrid','on','xgrid','on');
            set(gca,'yminorgrid','on')
            set(gca,'fontsize',12)
            xlabel('frequency [Hz]');
            ylabel('Power [10*log10 \muV^2/Hz]');
            ylimits1=get(gca,'ylim');
            title({['noise frequency: ' num2str(noisefreq,'%4.2f') 'Hz'],['raw ratio of noise to surroundings: ' num2str(ratioNoiseRaw,'%4.2f')]})
            box off
            
            
            % plot removed and clean spectrum
            
            pos = [15:18 25:28];
            ax2 = subplot(60,10,[pos+60*4 pos+60*5 pos+60*6 pos+60*7 pos+60*8 pos+60*9]);
            hold on
            
            %             plot(f/(f_noise*srate),pxx_removed_log,'color',[0.95 0.85 0.75]);
            %             plot(f/(f_noise*srate),pxx_clean_log,'color',[0.7 0.8 0.82]);
            removedhandle = plot(f/(f_noise*srate),mean(pxx_removed_log,2),'color',red,'linewidth',1.5);
            cleanhandle = plot(f/(f_noise*srate),mean(pxx_clean_log,2),'color',green,'linewidth',1.5);
            
            
            % adjust plot
            set(gca,'ygrid','on','xgrid','on');
            set(gca,'yminorgrid','on')
            set(gca,'fontsize',12)
            set(gca,'yticklabel',[]); ylabel([]);
            xlabel('frequency relative to noise [Hz]');
            ylimits2=get(gca,'ylim');
            ylimits(1)=min(ylimits1(1),ylimits2(1)); ylimits(2)=max(ylimits1(2),ylimits2(2));
            title({['removed power at ' num2str(noisefreq,'%4.2f') 'Hz: ' num2str(proportionRemovedNoise*100,'%4.2f') '%']
                ['cleaned ratio of noise to surroundings: ' num2str(ratioNoiseClean,'%4.2f')]})
            
            ylim(ax1,ylimits);
            ylim(ax2,ylimits);
            xlim(ax1,[min(f)-max(f)*0.0032 max(f)]);
            xlim(ax2,[min(f/(f_noise*srate))-max(f/(f_noise*srate))*0.003 max(f/(f_noise*srate))]);
            
            box off
            
            % plot shaded min max freq areas
            
            freqhandles = fill(ax1,[0 minfreq minfreq 0],[ylimits(1) ylimits(1) ylimits(2) ylimits(2)],[0 0 0],'facealpha',0.1,'edgealpha',0);
            fill(ax1,[maxfreq max(f) max(f) maxfreq],[ylimits(1) ylimits(1) ylimits(2) ylimits(2)],[0 0 0],'facealpha',0.1,'edgealpha',0)
            %             legend(ax1,[meanhandles singlehandles(1) freqhandles],{'raw data (mean)','raw data (single channels)','below min / above max freq'},'edgecolor',[0.8 0.8 0.8]);
            legend(ax1,[meanhandles freqhandles],{'raw data','below min / above max freq'},'edgecolor',[0.8 0.8 0.8]);
            
            fill(ax2,[0 minfreq minfreq 0]/noisefreq,[ylimits(1) ylimits(1) ylimits(2) ylimits(2)],[0 0 0],'facealpha',0.1,'edgealpha',0)
            fill(ax2,[maxfreq max(f) max(f) maxfreq]/noisefreq,[ylimits(1) ylimits(1) ylimits(2) ylimits(2)],[0 0 0],'facealpha',0.1,'edgealpha',0)
            legend(ax2,[cleanhandle,removedhandle],{'clean data','removed data'},'edgecolor',[0.8 0.8 0.8]);
            
            
            % plot below noise
            pos = [];
            for i = 26:57
                pos = [pos i*40-5:i*40];
            end
            subplot(60,40,pos);
            
            plot(f(this_freq_idx_belownoise),mean(pxx_raw_log(this_freq_idx_belownoise,:),2),'color',grey,'linewidth',1.5);
            hold on
            plot(f(this_freq_idx_belownoise),mean(pxx_clean_log(this_freq_idx_belownoise,:),2),'color',green,'linewidth',1.5);
            legend({'raw data','clean data'},'edgecolor',[0.8 0.8 0.8]);
            set(gca,'ygrid','on','xgrid','on');
            set(gca,'yminorgrid','on')
            set(gca,'fontsize',12)
            xlabel('frequency [Hz]');
            box off
            xlim([min(f(this_freq_idx_belownoise)) max(f(this_freq_idx_belownoise))]);
            title({['total power removed: ' num2str(proportionRemoved*100,'%4.2f') '%']
                [num2str(noisefreq-11,'%4.0f') ' - ' num2str(noisefreq-1,'%4.0f') 'Hz power removed: ' num2str(proportionRemovedBelowNoise*100,'%4.2f') '%']})
            
            drawnow
            
            %%
        end
        
        % decide if redo cleaning (plot needs to be before because it shows incorrect sigma otherwise)
        
        cleaningDone = 1;
        
        if adaptiveNremove && adaptiveSigma
            if cleaningTooStong && thisZaplineConfig.noiseCompDetectSigma < maxSigma
                cleaningTooStongOnce = 1;
                thisZaplineConfig.noiseCompDetectSigma = min(thisZaplineConfig.noiseCompDetectSigma + 0.25,maxSigma);
                cleaningDone = 0;
                thisFixedNremove = max(thisFixedNremove-1,fixedNremove);
                disp(['Cleaning too strong! Increasing sigma for noise component detection to '...
                    num2str(thisZaplineConfig.noiseCompDetectSigma) ' and setting minimum number of removed components to '...
                    num2str(thisFixedNremove) '.'])
                continue
            end
            
            % cleaning must never have been too strong, this is to ensure minimal impact on the spectrum other than
            % noise freq
            if cleaningTooWeak && ~cleaningTooStongOnce && thisZaplineConfig.noiseCompDetectSigma > minSigma
                thisZaplineConfig.noiseCompDetectSigma = max(thisZaplineConfig.noiseCompDetectSigma - 0.25,minSigma);
                cleaningDone = 0;
                thisFixedNremove = thisFixedNremove+1;
                disp(['Cleaning too weak! Reducing sigma for noise component detection to '...
                    num2str(thisZaplineConfig.noiseCompDetectSigma) ' and setting minimum number of removed components to '...
                    num2str(thisFixedNremove) '.'])
            end
        end
    end
    
    data = cleanData;
    resScores(i_noisefreq,1:size(scores,1),1:size(scores,2)) = scores;
    resNremoveFinal(i_noisefreq,1:size(NremoveFinal,1),1:size(NremoveFinal,2)) = NremoveFinal;
    resNoisePeaks(i_noisefreq,1:size(noisePeaks,1),1:size(noisePeaks,2)) = noisePeaks;
    resFoundNoise(i_noisefreq,1:size(foundNoise,1),1:size(foundNoise,2)) = foundNoise;
    resSigmaFinal(i_noisefreq) = thisZaplineConfig.noiseCompDetectSigma;
    resProportionRemoved(i_noisefreq) = proportionRemoved;
    resProportionRemovedNoise(i_noisefreq) = proportionRemovedNoise;
    resProportionRemovedBelowNoise(i_noisefreq) = proportionRemovedBelowNoise;
    resRatioNoiseRaw(i_noisefreq) = ratioNoiseRaw;
    resRatioNoiseClean(i_noisefreq) = ratioNoiseClean;
    resProportionBelowLower(i_noisefreq) = proportionBelowLower;
    resProportionAboveUpper(i_noisefreq) = proportionAboveUpper;
    
    if automaticFreqDetection
        disp(['Searching for first noise frequency between ' num2str(noisefreqs(i_noisefreq)+detailedFreqBoundsUpper(2)) ' and ' num2str(maxfreq) 'Hz...'])
        
        [nextfreq,~,~,thresh] = find_next_noisefreq(pxx_clean_log,f,...
            noisefreqs(i_noisefreq)+detailedFreqBoundsUpper(2),coarseFreqDetectPowerDiff,detectionWinsize,maxfreq,...
            coarseFreqDetectLowerPowerDiff,verbose);
        if ~isempty(nextfreq)
            noisefreqs(end+1)=nextfreq;
        end
    end
    i_noisefreq = i_noisefreq + 1;
    
end

% add flat channels back in
if ~isempty(flat_channels_idx)
    warning(['Detected flat channels were ignored and are added back in after Zapline plus processing: ' num2str(flat_channels_idx)])
    
    fullCleanData = [];
    
    i_last = 1;
    i_last_clean = 1;
    
    for i_flatchan = 1:length(flat_channels_idx)
        
        flatchan = flat_channels_idx(i_flatchan);
        fullCleanData(:,i_last:flatchan-1) = cleanData(:,i_last_clean:flatchan-i_flatchan);
        fullCleanData(:,flatchan) = flat_channels_data(:,i_flatchan);
        
        i_last = flatchan+1;
        i_last_clean = flatchan-i_flatchan+1;
        
    end
    
    fullCleanData(:,i_last:size(cleanData,2)+length(flat_channels_idx)) = cleanData(:,i_last_clean:end);
        
    cleanData = fullCleanData;
    
end

if transposeData
    cleanData = cleanData';
end

if plotResults && ~exist('plothandles','var')
    
    figThis = figBase+1;
    plothandles(i_noisefreq) = figure(figThis);
    clf; 
    set(gcf,'Color','w','InvertHardCopy','off', 'units','normalized','outerposition',[0.2 0.2 0.7 0.7])
    set(gcf,'name','No noise')
    
    grey = [0.2 0.2 0.2];
    plot(f,mean(pxx_raw_log,2),'color',grey);
    legend({'raw'},'edgecolor',[0.8 0.8 0.8]);
    set(gca,'ygrid','on','xgrid','on');
    set(gca,'yminorgrid','on')
    xlabel('frequency');
    ylabel('Power [10*log10 \muV^2/Hz]');
    title('no noise found')
    box off
    xlim([min(f)-max(f)*0.0015 max(f)]);
    
    
end

if ~exist('plothandles','var')
    plothandles = [];
end

zaplineConfig.noisefreqs = noisefreqs;

if saveSpectra
    analyticsResults.cleanSpectrumLog = pxx_clean_log;
end
analyticsResults.sigmaFinal = resSigmaFinal;
analyticsResults.proportionRemoved = resProportionRemoved;
analyticsResults.proportionRemovedNoise = resProportionRemovedNoise;
analyticsResults.proportionRemovedBelowNoise = resProportionRemovedBelowNoise;
analyticsResults.proportionBelowLower = resProportionBelowLower;
analyticsResults.proportionAboveUpper = resProportionAboveUpper;
analyticsResults.ratioNoiseRaw = resRatioNoiseRaw;
analyticsResults.ratioNoiseClean = resRatioNoiseClean;
analyticsResults.NremoveFinal = resNremoveFinal;
analyticsResults.scores = resScores;
analyticsResults.noisePeaks = resNoisePeaks;
analyticsResults.foundNoise = resFoundNoise;

disp('Cleaning with ZapLine-plus done!')
