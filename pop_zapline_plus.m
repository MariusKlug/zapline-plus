% Author: Arnaud Delorme

function [EEG, com] = pop_zapline_plus(EEG, varargin)
com = '';

if nargin < 2
        cb_auto = [ 'set(findobj(gcbf, ''userdata'', ''comp''), ''enable'', fastif(get(gcbo, ''value''), ''off'', ''on''));' ...
                    'if get(gcbo, ''value''), set(findobj(gcbf, ''tag'', ''comp''), ''string'', ''''); else set(findobj(gcbf, ''tag'', ''comp''), ''string'', ''3''); end' ];
                    
        promptstr    = { ...
        { 'style'  'text'       'string' 'Enter line noise frequency ("line" searches for 50/60 Hz line noise, empty is auto detection 17-99 Hz):' } ...
        { 'style'  'edit'       'string' 'line' 'tag' 'freq' } ...
        { 'style'  'text'       'string' 'Enter auto noise detector threshold (lower detects more, recommended 3 to 10):' } ...
        { 'style'  'edit'       'string' '4' 'tag' 'detectorthreshold' } ...
        { 'style'  'checkbox'   'string' 'Divide data in chunks' 'value' 1 'tag' 'chunks' } ...
        { 'style'  'checkbox'   'string' 'Automatically detect the number of noise components' 'value' 1 'tag' 'autodetect' 'callback' cb_auto } ...
        { 'style'  'text'       'string' 'Or enter the number of components:' 'userdata' 'comp'  'enable' 'off'} ...
        { 'style'  'edit'       'string' '' 'tag' 'comp' 'userdata' 'comp' 'enable' 'off' } ...
        { 'style'  'checkbox'   'string' 'Plot results' 'value' 1 'tag' 'plot' } ...
        };
    geometry = {[2 0.5] [2 0.5] [1] [1] [2 0.5] [1] };
    
    [~,~,~,res] = inputgui( 'geometry', geometry, 'geomvert', [1 1 1 1 1 1], 'uilist', promptstr, 'helpcom', 'pophelp(''pop_zapline_plus'')', 'title', 'Process data with Zapline-plus');
    if isempty(res), return; end
    
    if strcmp(res.freq,'line'),             options = { 'noisefreqs' 'line'}; else options = { 'noisefreqs' str2num( [ '[' res.freq ']' ] )}; end
    if ~isempty(res.detectorthreshold),     options = [ options { 'coarseFreqDetectPowerDiff' str2num( [ '[' res.detectorthreshold ']' ] ) } ]; end
    if res.chunks,                          options = [ options { 'chunkLength' 0 } ]; else options = [ options { 'chunkLength' 1e9 } ]; end
    if res.autodetect,                      options = [ options { 'adaptiveNremove' 1 'fixedNremove' 1 } ]; else options = [ options { 'adaptiveNremove' 0 'fixedNremove' str2double(res.comp) } ]; end
    if ~res.plot,                           options = [ options { 'plotResults' 0 } ]; end
else
    options = varargin;
end

EEG = clean_data_with_zapline_plus_eeglab_wrapper(EEG, struct(options{:})); %, param);

com = sprintf('EEG = pop_zapline_plus(EEG, %s);', vararg2str(options));
