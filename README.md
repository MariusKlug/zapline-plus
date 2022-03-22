# Zapline-plus
Improvements of the ZapLine function to remove line noise from EEG/MEG data. Adds automatic detection of the number of components to remove, and chunks the data into segments to account for nonstationarities.

Dependencies of Noisetools are provided with permission by Alain de Cheveigné. Please visit the original repository for more info and additional noise removal tools: http://audition.ens.fr/adc/NoiseTools/

# Quick start
```matlab
cleanedData = clean_data_with_zapline_plus(data,srate);
```

Or if you live in the EEGlab universe:
```matlab
EEG = clean_data_with_zapline_plus_eeglab_wrapper(EEG,struct('noisefreqs','line')) % specifying the config is optional
```

# Please cite

Klug, M., & Kloosterman, N. A. (2022).Zapline-plus: A Zapline extension for automatic and adaptiveremoval of frequency-specific noise artifacts in M/EEG. Human Brain Mapping,1–16. https://doi.org/10.1002/hbm.25832

de Cheveigne, A. (2020) ZapLine: a simple and effective method to remove power line artifacts. NeuroImage, 1, 1-13. https://doi.org/10.1016/j.neuroimage.2019.116356

