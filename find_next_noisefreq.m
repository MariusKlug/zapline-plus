function [noisefreq,thisfreqs,thisdata,threshfound] = find_next_noisefreq(pxx,f,minfreq,threshdiff,winsizeHz,maxfreq,lower_threshdiff,verbose)
% simple helper which searches the next noise frequency based on the spectrum starting from a minimum frequency. uses a
% moving window to find outliers defined as being above the center power (mean of left and right thirds around the
% current frequency). expects input to be in log space, so a threshold being linearly above the center power corresponds
% to a multiplication of the center power as the threshold.

if ~exist('minfreq')|isempty(minfreq)
    minfreq = 0;
end
if ~exist('threshdiff')|isempty(threshdiff)
    threshdiff = 5;
end
if ~exist('winsizeHz')|isempty(winsizeHz)
    winsizeHz = 3;
end
if ~exist('maxfreq')|isempty(maxfreq)
    maxfreq = max(f)*0.85;
end
if ~exist('lower_threshdiff')|isempty(lower_threshdiff)
    lower_threshdiff = 1.76091259055681; % 10^0.176091259055681 = 1.5 -> 1.5x center
end
if ~exist('verbose')|isempty(verbose)
    verbose = 0;
end

if verbose
    disp(['Searching for first noise freq between ' num2str(minfreq) 'Hz and ' num2str(maxfreq) 'Hz...'])
end

noisefreq = [];
threshfound = [];
thisfreqs = [];
thisdata = [];
winsize = round(size(pxx,1)/(max(f) - min(f))*winsizeHz);
meandata = mean(pxx,2);

detectionstart = 0;
detected = 1;
i_startdetected  = 0;
i_enddetected  = 0;

i_start = max(find(f>minfreq,1,'first')+1,round(winsize/2));
i_end = min(find(f<maxfreq,1,'last'),length(f)-round(winsize/2));

lastfreq = 0;
% detection left and right of freq of interest
for i = i_start-round(winsize/2):i_end-round(winsize/2)+1
    
    thisdata = meandata(i:i+winsize-1);
    thisfreqs = f(i:i+winsize-1);
    
    thisfreq = round(thisfreqs(round(end/2)));
    if verbose && thisfreq > lastfreq
        fprintf('%g,',thisfreq)
        lastfreq = thisfreq;
    end
    
    
    
    third = round(length(thisdata)/3);
    center_thisdata = mean(thisdata([1:third third*2:end]));
    thresh = center_thisdata + threshdiff; % in log space this corresponds to a multiple of the center
    
    % old detector tests left in for the record
    %     mean_lower_quantile_thisdata = mean([quantile(thisdata(1:third),0.05) quantile(thisdata(third*2:end),0.05)]);
    %     thresh = min(center_thisdata + threshdiff * (center_thisdata -
    %     mean_lower_quantile_thisdata),center_thisdata+10); % 10 times more than center is minimum for noise detect in
    %     case some noise behaves weirdly
    
    %     thresh = threshmult*1.4826*mad(thisdata,1)+median(thisdata); thresh = median(thisdata) *...
    %                 (threshmult * median(thisdata) / quantile(thisdata,0.05));
    
    %     quantileratio = quantile(thisdata,0.95) / quantile(thisdata,0.05); interquantilecenter =
    %     sqrt((quantile(thisdata,0.95) * quantile(thisdata,0.05)));
    
    %     thresh = interquantilecenter * (1.6*quantileratio);
    
    % for test purposes
    %     figure; plot(thisfreqs,thisdata) hold on; plot(xlim,[thresh thresh],'k') plot(xlim,[center_thisdata
    %     center_thisdata],'k') plot(xlim,[mean_lower_quantile_thisdata mean_lower_quantile_thisdata],'k')
    
    if ~detected
        detectednew = thisdata(round(end/2))>thresh;
        if detectednew
            i_startdetected = round(i+(winsize-1)/2);
            threshfound = thresh;
        end
    else
        detectednew = thisdata(round(end/2))>center_thisdata+lower_threshdiff;
        i_enddetected = round(i+(winsize-1)/2);
    end
    
    if ~detectionstart && detected && ~detectednew
        detectionstart = 1;
    elseif detectionstart && detected && ~detectednew
        
        noisefreq=f(meandata == max(meandata(i_startdetected:i_enddetected)));
        if verbose
            fprintf('found %gHz!\n',noisefreq)
            figure; plot(thisfreqs,thisdata)
            hold on; plot(xlim,[thresh thresh],'r')
            plot(xlim,[center_thisdata center_thisdata],'k')
            title(num2str(noisefreq))
        end
        return
    end
    
    detected = detectednew;
    
end
if verbose
    fprintf('none found.\n',noisefreq)
end