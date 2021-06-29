% CLEAN_DATA_WITH_ZAPLINE - Removial of frequency artifacts using ZapLine to remove noise from EEG/MEG data. Adds
% automatic detection of the noise frequencies as well as number of components to remove, and chunks the data into
% segments to account for nonstationarities. Based on: de Cheveigne, A. (2020) ZapLine: a simple and effective method to
% remove power line artifacts. Neuroimage, 1, 1-13.
%
% Requires noisetools to be installed: http://audition.ens.fr/adc/NoiseTools/
%
% Usage:
%   >>  [cleanData, resNremoveFinal, resScores, plothandles] = clean_data_with_zapline(data, srate, varargin);
%
% Required Inputs:
%   data                    - MEEG data matrix
%   srate                   - sampling rate in Hz
%
% Optional Parameters:
%   noisefreqs               - vector with one or more noise frequencies to be removed. if empty or missing, noise freqs
%                               will be detected automatically
%   minfreq                 - minimum frequency to be considered as noise when searching for noise freqs automatically.
%                               (default = 13)
%   freqDetectMult          - multiplier for the median absolute deviation detector of the noise frequency (def = 3.5)
%   freqDetectMultFine      - multiplier for the median absolute deviation detector of the fine noise frequency detection (def = 1.75)
%   adaptiveNremove         - bool. if automatic adaptation of removal should be used. (default = 1)
%   fixedNremove            - fixed number of removed components. if adaptive removal is used, this
%                               will be the minimum. (default = 0)
%   chunkLength             - numerical. length of chunks to be cleaned in seconds. if set to 0, no chunks will be used.
%                               (default = 30)
%   plotResults             - bool. if plot should be created. takes time to compute the spectrum. (default = 1)
%   figBase                 - integer. figure number to be created and plotted in. each iteration of noisefreqs increases
%                               this number by 1. (default = 100)
%   nkeep                   - integer. PCA reduction of components before removal. (default = number of channels)
%   noiseCompDetectSigma    - numerical. iterative outlier detection sigma threshold. (default = 3)
%
% Outputs:
%   cleanData               - clean EEG data matrix
%   zaplineNremoveFinal     - matrix of number of removed components per noisefreq and chunk
%   scores                  - matrix of artifact component scores per noisefreq and chunk
%   zaplineConfig           - config struct with all used parameters
%   plothandles             - vector of handles to the created figures
%
% Example:
%   [cleanData, resNremoveFinal, resScores, plothandles] = clean_data_with_zapline(EEG.data,EEG.srate);
%   [cleanData, resNremoveFinal, resScores, plothandles] = clean_data_with_zapline(EEG.data,EEG.srate,'adaptiveSigma',0,'chunkLength',200);
%
% See also:
%   nt_zapline_plus, iterative_outlier_removal
%
% Author: Marius Klug, 2021

function [cleanData, resNremoveFinal, resScores, zaplineConfig, plothandles] = clean_data_with_zapline(data, srate, varargin)

if nargin == 0
    help clean_data_with_zapline
    return
end

disp('Removing frequency artifacts using ZapLine with adaptations for automatic component selection and chunked data.')
disp('---------------- PLEASE CITE ------------------')
disp(' ')
disp('de Cheveigne, A. (2020) ZapLine: a simple and effective method to remove power line artifacts. NeuroImage, 1, 1-13.')
disp(' ')
disp('---------------- PLEASE CITE ------------------')

% input parsing settings
p = inputParser;
p.CaseSensitive = false;

addRequired(p, 'data', @(x) validateattributes(x,{'numeric'},{'2d'},'clean_EEG_with_zapline','data'))
addRequired(p, 'srate', @(x) validateattributes(x,{'numeric'},{'positive','scalar','integer'},'clean_EEG_with_zapline','srate'))
addOptional(p, 'noisefreqs', [], @(x) validateattributes(x,{'numeric'},{'positive','vector'},'clean_EEG_with_zapline','noisefreqs'))
addOptional(p, 'fixedNremove', 1, @(x) validateattributes(x,{'numeric'},{'integer','scalar'},'clean_EEG_with_zapline','fixedNremove'));
addOptional(p, 'minfreq', 13, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','minfreq'))
addOptional(p, 'maxfreq', 99, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','maxfreq'))
addOptional(p, 'detectionWinsize', 6, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','detectionWinsize'))
addOptional(p, 'coarseFreqDetectPowerDiff', 4, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','coarseFreqDetectPowerDiff'))
addOptional(p, 'coarseFreqDetectLowerPowerDiff', 1.76091259055681, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','coarseFreqDetectLowerPowerDiff'))
addOptional(p, 'freqDetectMultFine', 2, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','freqDetectMultFine'))
addOptional(p, 'maxProportionAboveUpper', 0.005, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','maxProportionAboveUpper'))
addOptional(p, 'maxProportionAboveLower', 0.005, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','maxProportionAboveLower'))
addOptional(p, 'adaptiveNremove', 1, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','adaptiveNremove'));
addOptional(p, 'noiseCompDetectSigma', 3, @(x) validateattributes(x,{'numeric'},{'scalar','positive'},'clean_EEG_with_zapline','noiseCompDetectSigma'));
addOptional(p, 'adaptiveSigma', 1, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','adaptiveSigma'));
addOptional(p, 'minsigma', 2.5, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','minsigma'))
addOptional(p, 'maxsigma', 4, @(x) validateattributes(x,{'numeric'},{'positive','scalar'},'clean_EEG_with_zapline','maxsigma'))
addOptional(p, 'chunkLength', 150, @(x) validateattributes(x,{'numeric'},{'scalar','integer'},'clean_EEG_with_zapline','chunkLength'));
addOptional(p, 'detailedFreqBoundsUpper', [-0.05 0.05], @(x) validateattributes(x,{'numeric'},{'vector'},'clean_EEG_with_zapline','detailedFreqBoundsUpper'))
addOptional(p, 'detailedFreqBoundsLower', [-0.4 0.1], @(x) validateattributes(x,{'numeric'},{'vector'},'clean_EEG_with_zapline','detailedFreqBoundsLower'))
addOptional(p, 'nkeep', 0, @(x) validateattributes(x,{'numeric'},{'scalar','integer','positive'},'clean_EEG_with_zapline','nkeep'));
addOptional(p, 'plotResults', 1, @(x) validateattributes(x,{'numeric','logical'},{'scalar','binary'},'clean_EEG_with_zapline','plotResults'));
addOptional(p, 'figBase', 100, @(x) validateattributes(x,{'numeric'},{'scalar','integer','positive'},'clean_EEG_with_zapline','figBase'));

% parse the input
parse(p,data,srate,varargin{:});

data = p.Results.data;
srate = p.Results.srate;
noisefreqs = p.Results.noisefreqs;
coarseFreqDetectPowerDiff = p.Results.coarseFreqDetectPowerDiff;
coarseFreqDetectLowerPowerDiff = p.Results.coarseFreqDetectLowerPowerDiff;
freqDetectMultFine = p.Results.freqDetectMultFine;
maxProportionAboveUpper = p.Results.maxProportionAboveUpper;
maxProportionAboveLower = p.Results.maxProportionAboveLower;
minfreq = p.Results.minfreq;
maxfreq = p.Results.maxfreq;
detectionWinsize = p.Results.detectionWinsize;
adaptiveNremove = p.Results.adaptiveNremove;
adaptiveSigma = p.Results.adaptiveSigma;
minSigma = p.Results.minsigma;
maxSigma = p.Results.maxsigma;
fixedNremove = p.Results.fixedNremove;
noiseCompDetectSigma = p.Results.noiseCompDetectSigma;
chunkLength = p.Results.chunkLength;
detailedFreqBoundsUpper = p.Results.detailedFreqBoundsUpper;
detailedFreqBoundsLower = p.Results.detailedFreqBoundsLower;
nkeep = p.Results.nkeep;
plotResults = p.Results.plotResults;
figBase = p.Results.figBase;

% finalize
transposeData = size(data,2)>size(data,1);
if transposeData
    data = data';
end
if nkeep == 0
    % our tests show actually better cleaning performance when no PCA reduction is used!
    
    %     nkeep = min(round(20+size(data,2)/4),size(data,2));
    %     disp(['Reducing the number of components to ' num2str(nkeep) ', set the ''nkeep'' flag to decide otherwise.'])
    nkeep = size(data,2);
end
if chunkLength == 0
    chunkLength = size(data,1)/srate;
end

zaplineConfig.adaptiveNremove = adaptiveNremove;
zaplineConfig.nkeep = nkeep;
zaplineConfig.fixedNremove = fixedNremove;
zaplineConfig.noiseCompDetectSigma = noiseCompDetectSigma;
zaplineConfig.chunkLength = chunkLength;
zaplineConfig.nkeep = nkeep;
zaplineConfig.noisefreqs = noisefreqs;
zaplineConfig.freqDetectMult = coarseFreqDetectPowerDiff;
zaplineConfig.freqDetectMultFine = freqDetectMultFine;

% default
cleanData = data;
resNremoveFinal = [];
resScores = [];

%% Clean each frequency one after another

i_noisefreq = 1;
automaticFreqDetection = isempty(noisefreqs);
if automaticFreqDetection
    disp('Computing initial spectrum...')
    % compute spectrum with frequency resolution of chunk length
    [pxx_raw,f]=pwelch(data,hanning(srate*chunkLength),[],[],srate);
    % log transform
    pxx_raw = 10*log10(pxx_raw);
    disp(['Searching for first noise frequency between ' num2str(minfreq) ' and ' num2str(maxfreq) 'Hz...'])
    verbose = 0;
    [noisefreqs,~,~,thresh]=find_next_noisefreq(pxx_raw,f,minfreq,coarseFreqDetectPowerDiff,detectionWinsize,maxfreq,...
        coarseFreqDetectLowerPowerDiff,verbose);
end

while i_noisefreq <= length(noisefreqs)
    
    noisefreq = noisefreqs(i_noisefreq);
    
    thisFixedNremove = fixedNremove;
    
    fprintf('Removing noise at %gHz... \n',noisefreq);
    
    figThis = figBase+i_noisefreq;
    
    cleaningDone = 0;
    cleaningTooStongOnce = 0;
    thisZaplineConfig = zaplineConfig;
    
    while ~cleaningDone
        
        % result data matrix
        cleanData = NaN(size(data));
        
        % last chunk must be larger than the others, to ensure fft works, at least 1 chunk must be used
        nChunks = max(floor(size(data,1)/srate/chunkLength),1);
        scores = NaN(nChunks,nkeep);
        NremoveFinal = NaN(nChunks,1);
        noisePeaks = NaN(nChunks,1);
        foundNoise = zeros(nChunks,1);
        
        for iChunk = 1:nChunks
            
            if mod(iChunk,round(nChunks/10))==0
                disp(['Chunk ' num2str(iChunk) ' of ' num2str(nChunks)])
            end
            
            if iChunk ~= nChunks
                chunkIndices = 1+chunkLength*srate*(iChunk-1):chunkLength*srate*(iChunk);
            else
                chunkIndices = 1+chunkLength*srate*(iChunk-1):size(data,1);
            end
            
            chunk = data(chunkIndices,:);
            
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
            
%             figure; plot(f,mean(pxx_chunk,2));
%             xlim([f(find(this_freq_idx,1,'first')) f(find(this_freq_idx,1,'last'))])
%             hold on
%             plot([f(find(this_freq_idx_detailed,1,'first')) f(find(this_freq_idx_detailed,1,'last'))],...
%                 [detailedNoiseThresh detailedNoiseThresh],'r')
%             plot(xlim,[center_thisdata center_thisdata])
%             plot(xlim,[mean_lower_quantile_thisdata mean_lower_quantile_thisdata])
%             title(['chunk ' num2str(iChunk) ', ' num2str(noisePeaks(iChunk))])
            
            
            if maxFinePower > detailedNoiseThresh
                % use adaptive cleaning
                foundNoise(iChunk) = 1;

                % needs to be normalized for zapline
                f_noise = noisePeaks(iChunk)/srate;
                [cleanData(chunkIndices,:),~,NremoveFinal(iChunk),thisScores] =...
                    nt_zapline_plus(chunk,f_noise,thisFixedNremove,thisZaplineConfig,0);

                scores(iChunk,1:length(thisScores)) = thisScores;

            else
                % no noise was found in chunk -> clean with fixed threshold to be sure (it might be a miss of the
                % detector), but use overall noisefreq
                noisePeaks(iChunk) = noisefreq;

                this_zaplineConfig_noNoise = thisZaplineConfig;
                this_zaplineConfig_noNoise.adaptiveNremove = 0;

                f_noise = noisePeaks(iChunk)/srate;
                [cleanData(chunkIndices,:),~,NremoveFinal(iChunk),thisScores] =...
                    nt_zapline_plus(chunk,f_noise,thisFixedNremove,this_zaplineConfig_noNoise,0);

                scores(iChunk,1:length(thisScores)) = thisScores;

            end
                        
%             [pxx_chunk,f]=pwelch(cleanData(chunkIndices,:),hanning(length(chunk)),[],[],srate);
%             pxx_chunk = 10*log10(pxx_chunk);
%             figure; plot(f,mean(pxx_chunk,2));
%             xlim([f(find(this_freq_idx,1,'first')) f(find(this_freq_idx,1,'last'))])
%             title(['chunk ' num2str(iChunk) ', ' num2str(noisePeaks(iChunk)) ', ' num2str(NremoveFinal(iChunk)) ' removed'])
            
        end
        disp('Done. Computing spectra...')
        
        % compute spectra
        [pxx_raw]=pwelch(data,hanning(srate*chunkLength),[],[],srate);
        pxx_raw = 10*log10(pxx_raw);
        [pxx_clean,f]=pwelch(cleanData,hanning(srate*chunkLength),[],[],srate);
        pxx_clean = 10*log10(pxx_clean);
        [pxx_removed]=pwelch(data-cleanData,hanning(srate*chunkLength),[],[],srate);
        pxx_removed = 10*log10(pxx_removed);
        
        proportion_removed = 10^((mean(pxx_removed(:))-mean(pxx_raw(:)))/10); % cause of log
        disp(['proportion of removed power: ' num2str(proportion_removed)]);
        
        
        % check if cleaning was too weak or too strong
        
        % determine center power by checking lower and upper third around noise freq, then check detailed lower and
        % upper threhsold. search area for weak is around the noisefreq, for strong its larger and a little below the
        % noisefreq because zapline makes a dent there
        thisFreqidx = f>noisefreq-(detectionWinsize/2) & f<noisefreq+(detectionWinsize/2);
        thisFreqidxUppercheck = f>noisefreq+detailedFreqBoundsUpper(1) & f<noisefreq+detailedFreqBoundsUpper(2);
        thisFreqidxLowercheck = f>noisefreq+detailedFreqBoundsLower(1) & f<noisefreq+detailedFreqBoundsLower(2);
        
        thisFineData = mean(pxx_clean(thisFreqidx,:),2);
        third = round(length(thisFineData)/3);
        centerThisData = mean(thisFineData([1:third third*2:end]));
        
        % measure of variation in this case is only lower quantile because upper quantile can be driven by spectral outliers
        meanLowerQuantileThisData = mean([quantile(thisFineData(1:third),0.05) quantile(thisFineData(third*2:end),0.05)]);
        
        remainingNoiseThreshUpper = centerThisData + freqDetectMultFine * (centerThisData - meanLowerQuantileThisData);
        remainingNoiseThreshLower = centerThisData - freqDetectMultFine * (centerThisData - meanLowerQuantileThisData);
        
        % if x% of the samples in the search area are below or above the thresh it's too strong or weak
        proportionAboveUpper = sum(mean(pxx_clean(thisFreqidxUppercheck,:),2) > remainingNoiseThreshUpper) / sum(thisFreqidxUppercheck);
        cleaningTooWeak =  proportionAboveUpper > maxProportionAboveUpper;
        
        proportionAboveLower = sum(mean(pxx_clean(thisFreqidxLowercheck,:),2) < remainingNoiseThreshLower) / sum(thisFreqidxLowercheck);
        cleaningTooStong = proportionAboveLower > maxProportionAboveLower;
        
        disp([num2str(round(proportionAboveUpper*100,2)) '% of frequency samples above thresh in the range of '...
            num2str(detailedFreqBoundsUpper(1)) ' to ' num2str(detailedFreqBoundsUpper(2)) 'Hz around noisefreq (threshold is '...
            num2str(maxProportionAboveUpper*100) '%).'])
        disp([num2str(round(proportionAboveLower*100,2)) '% of frequency samples below thresh in the range of '...
            num2str(detailedFreqBoundsLower(1)) ' to ' num2str(detailedFreqBoundsLower(2)) 'Hz around noisefreq (threshold is '...
            num2str(maxProportionAboveLower*100) '%).'])
        
        if plotResults
            
            %%
            red = [230 100 50]/256;
            green = [0 97 100]/256;
            grey = [0.2 0.2 0.2];
            
            this_freq_idx_plot = f>=noisefreq-1.1 & f<=noisefreq+1.1;
            plothandles(i_noisefreq) = figure(figThis);
            clf; set(gcf,'color','w','Position',[31 256 1030 600])
            
            % plot original power
            subplot(3,30,[1:5]);
            
            plot(f(this_freq_idx_plot),mean(pxx_raw(this_freq_idx_plot,:),2),'color',grey)
            xlim([f(find(this_freq_idx_plot,1,'first'))-0.01 f(find(this_freq_idx_plot,1,'last'))])
            ylimits = ylim;
            if automaticFreqDetection
                ylim([min(mean(pxx_raw(this_freq_idx_plot,:),2))-0.5 thresh+1])
            end
            box off
            
            hold on
            if automaticFreqDetection
                plot(xlim,[thresh thresh],'color',red)
            end
            xlabel('frequency')
            ylabel('mean(10*log10(psd))')
            title({'detected frequency:', [num2str(noisefreq) 'Hz']})
            
            % plot nremoved
            pos = 8:17;
            subplot(24,60,[pos pos+30]*2-1);
            
            plot(NremoveFinal,'color',grey)
            xlim([0.8 length(NremoveFinal)])
            ylim([0 max(NremoveFinal)+1])
            title({['# removed comps per ' num2str(chunkLength)...
                's chunk, \mu = ' num2str(round(mean(NremoveFinal),2))]})
            hold on
            foundNoisePlot = foundNoise;
            foundNoisePlot(foundNoisePlot==1) = NaN;
            foundNoisePlot(~isnan(foundNoisePlot)) = NremoveFinal(~isnan(foundNoisePlot));
            plot(foundNoisePlot,'o','color',green);
            box off
            
             % plot noisepeaks
            subplot(24*2,60,[pos+30*9 pos+30*10 pos+30*11 pos+30*12]*2-1); % lol dont judge me it works
            
            plot(noisePeaks,'color',grey)
            xlim([0.8 length(NremoveFinal)])
            maxdiff = max([(max(noisePeaks))-noisefreq noisefreq-(min(noisePeaks))]);
            if maxdiff == 0
                maxdiff = 0.01;
            end
            ylim([noisefreq-maxdiff*1.5 noisefreq+maxdiff*1.5])
            xlabel('chunk')
            title({['individual noise frequencies [Hz]']})
            hold on
            foundNoisePlot = foundNoise;
            foundNoisePlot(foundNoisePlot==1) = NaN;
            foundNoisePlot(~isnan(foundNoisePlot)) = noisePeaks(~isnan(foundNoisePlot));
            linehandle = plot(foundNoisePlot,'o','color',green);
            l = legend(linehandle,'no clear noise peak found','edgecolor',[0.8 0.8 0.8],'position',...
                [0.368446603160784 0.806110653943486 0.129126211219621 0.026666666070620]);
            box off
            
            % plot scores
            subplot(3,30,[19:23]);
            
            plot(nanmean(scores,1),'color',grey)
            hold on
            plot([mean(NremoveFinal)+1 mean(NremoveFinal)+1],ylim,'color',red)
            xlim([0.7 round(size(scores,2)/3)])
            title({'mean artifact scores [a.u.]', ['\sigma for detection = ' num2str(thisZaplineConfig.noiseCompDetectSigma)]})
            xlabel('component')
            box off
            
            % plot new power
            subplot(3,30,[26:30]);
            
            plot(f(this_freq_idx_plot),mean(pxx_clean(this_freq_idx_plot,:),2),'color', green)
            
            xlim([f(find(this_freq_idx_plot,1,'first'))-0.01 f(find(this_freq_idx_plot,1,'last'))])
            hold on
            
            l1 = plot([f(find(thisFreqidxUppercheck,1,'first')) f(find(thisFreqidxUppercheck,1,'last'))],...
                [remainingNoiseThreshUpper remainingNoiseThreshUpper],'color',grey);
            l2 = plot([f(find(thisFreqidxLowercheck,1,'first')) f(find(thisFreqidxLowercheck,1,'last'))],...
            [remainingNoiseThreshLower remainingNoiseThreshLower],'color',red);
            ylim([remainingNoiseThreshLower-0.25*(remainingNoiseThreshUpper-remainingNoiseThreshLower)
                remainingNoiseThreshUpper+(remainingNoiseThreshUpper-remainingNoiseThreshLower)])
            
            
            xlabel('frequency')
            ylabel('mean(10*log10(psd))')
            title('cleaned spectrum')
            legend([l1 l2], {[num2str(round(proportionAboveUpper*100,2)) '% above']
                [num2str(round(proportionAboveLower*100,2)) '% below']},...
                'location','north','edgecolor',[0.8 0.8 0.8])
            box off
            
            
            % plot starting spectrum
            
            pos = [11:15 21:25];
            ax1 = subplot(60,10,[pos+60*4 pos+60*5 pos+60*6 pos+60*7 pos+60*8 pos+60*9]);
            
            plot(f,mean(pxx_raw,2));
            legend('raw','edgecolor',[0.8 0.8 0.8]); 
            set(gca,'ygrid','on','xgrid','on');
            set(gca,'yminorgrid','on')
            xlabel('frequency');
            ylabel('mean(10*log10(psd))');
            yl1=get(gca,'ylim');
            hh=get(gca,'children');
            set(hh(1),'color',grey)
            title(['noise frequency: ' num2str(noisefreq) 'Hz'])
            box off
            
            % plot removed noise spectrum
            
            pos = [16:20 26:30];
            subplot(60,10,[pos+60*4 pos+60*5 pos+60*6 pos+60*7 pos+60*8 pos+60*9]); 
            
            plot(f/(f_noise*srate),mean(pxx_removed,2));
            
            % plot clean spectrum
            hold on
            
            pos = [16:20 26:30];
            ax2 = subplot(60,10,[pos+60*4 pos+60*5 pos+60*6 pos+60*7 pos+60*8 pos+60*9]);
            
            plot(f/(f_noise*srate),mean(pxx_clean,2));
            
            % adjust plot
            legend('removed','clean','edgecolor',[0.8 0.8 0.8]); 
            set(gca,'ygrid','on','xgrid','on');
            set(gca,'yminorgrid','on')
            set(gca,'yticklabel',[]); ylabel([]);
            xlabel('frequency (relative to noise)');
            yl2=get(gca,'ylim');
            hh=get(gca,'children');
            set(hh(2),'color',red); set(hh(1), 'color', green);
            yl(1)=min(yl1(1),yl2(1)); yl(2)=max(yl1(2),yl2(2));
            title(['proportion removed: ' num2str(proportion_removed)])
            
            ylim(ax1,yl);
            ylim(ax2,yl);
            xlim(ax1,[min(f)-max(f)*0.0032 max(f)]);
            xlim(ax2,[min(f/(f_noise*srate))-max(f/(f_noise*srate))*0.004 max(f/(f_noise*srate))]);
            
            box off
            drawnow
            
            %%
        end
        
        % decide if redo cleaning (plot needs to be before because it shows incorrect sigma otherwise)
        
        cleaningDone = 1;
        
        if adaptiveSigma
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
    
    if automaticFreqDetection
        disp(['Searching for first noise frequency between ' num2str(noisefreqs(i_noisefreq)+detailedFreqBoundsUpper(2)) ' and ' num2str(maxfreq) 'Hz...'])
        [nextfreq,~,~,thresh] = find_next_noisefreq(pxx_clean,f,...
            noisefreqs(i_noisefreq)+detailedFreqBoundsUpper(2),coarseFreqDetectPowerDiff,detectionWinsize,maxfreq,...
            coarseFreqDetectLowerPowerDiff,verbose);
        if ~isempty(nextfreq)
            noisefreqs(end+1)=nextfreq;
        end
    end
    i_noisefreq = i_noisefreq + 1;
    
end

if transposeData
    cleanData = cleanData';
end

if ~exist('plothandles','var')
    plothandles = [];
end

zaplineConfig.noisefreqs = noisefreqs;

disp('Cleaning with ZapLine done!')