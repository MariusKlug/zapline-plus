# Zapline-plus
Improvements of the ZapLine function to remove line noise from EEG/MEG data. Adds automatic detection of the number of components to remove, and chunks the data into segments to account for nonstationarities.

Dependencies of Noisetools are provided with permission by Alain de Cheveigné. Please visit the original repository for more info and additional noise removal tools: http://audition.ens.fr/adc/NoiseTools/

# Quick start
```matlab
cleanedData = clean_data_with_zapline_plus(data,srate);
```

Or if you live in the EEGlab universe:
```matlab
EEG = clean_data_with_zapline_plus_eeglab_wrapper(EEG,struct('noisefreqs',[50])) % specifying the config is optional
```

# Please cite

Original Zapline paper: Cheveigné, Alain de. 2020. “ZapLine: A Simple and Effective Method to Remove Power Line Artifacts.” NeuroImage 207 (February): 116356. https://www.sciencedirect.com/science/article/pii/S1053811919309474

Zapline-plus paper: Klug, M., and N. A. Kloosterman. 2021. “Zapline-plus: A Zapline Extension for Automatic and Adaptive Removal of Frequency-Specific Noise Artifacts in M/EEG.” bioRxiv. https://www.biorxiv.org/content/10.1101/2021.10.18.464805.abstract.

