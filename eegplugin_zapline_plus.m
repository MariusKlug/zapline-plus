% eegplugin_zapline_plus() - EEGLAB plugin for Zapline-plus to remove
%                            
% Usage:
%   >> eegplugin_zapline_plus(fig, trystrs, catchstrs);
%
% Inputs:
%   fig        - [integer]  EEGLAB figure
%   trystrs    - [struct] "try" strings for menu callbacks.
%   catchstrs  - [struct] "catch" strings for menu callbacks.

function vers = eegplugin_zapline_plus(fig, trystrs, catchstrs)

    vers = '1.0';
    if nargin < 3
        error('eegplugin_zapline_plus requires 3 arguments');
    end
    
    % add folder to path
    % ------------------
    if ~exist('eegplugin_zapline_plus')
        p = which('eegplugin_zapline_plus.m');
        p = p(1:findstr(p,'eegplugin_zapline_plus.m')-1);
        addpath( p );
    end
    
    % find import data menu
    % ---------------------
    menui1 = findobj(fig, 'tag', 'tools');
    
    % menu callbacks
    % --------------
    comcnt1 = [ trystrs.no_check '[EEG,LASTCOM] = pop_zapline_plus(EEG); '  catchstrs.new_and_hist ];
                
    % create menus
    % ------------
    uimenu( menui1, 'label', 'Zapline plus noise removal', 'separator', 'on', 'callback', comcnt1);
