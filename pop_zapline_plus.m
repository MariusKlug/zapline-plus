function [EEG, com] = pop_zapline_plus(EEG, zaplineConfig)
% Super small wrapper for zapline-plus that takes an EEGLAB EEG struct
com = '';

if nargin < 2
    
        promptstr    = { ...
        { 'style'  'text'       'string' 'Enter parameter:' } ...
        { 'style'  'edit'       'string' '' 'tag' 'param' } ...
        };
    geometry = {[2 1.5] };
    
    [~,~,~,res] = inputgui( 'geometry', geometry, 'geomvert', [1], 'uilist', promptstr, 'helpcom', 'pophelp(''pop_zapline_plus'')', 'title', 'Process data with Zapline-plus');
    if isempty(res), return; end
    
    param = res.param;
end

EEG = clean_data_with_zapline_plus_eeglab_wrapper(EEG); %, param);

com = sprintf('[STUDY, ALLEEG] = pop_zapline_plus(EEG, %s);', vararg2str(param));
