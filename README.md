# Southern-Ocean-polynyas-in-CMIP6-models
This code was used for analysis of polynyas (areas, location, frequency) in CMIP6 models. It was written by Martin Mohrmann. 
Further explanations can be found in the corresponding paper "Southern Ocean polynyas in CMIP6 models", accepted for publication in the Cryosphere in Jul 2021.
Authors: Martin Mohrmann, Céline Heuzé, Sebastiaan Swart

If you have any questions about or interest in the provided codes, please feel free to contact us.

The algorithm to find polynyas is implemented in polynyaareas_remastered.py. It includes one function to analyse observational data and one function to analyse the different CMIP6 models. As input files, it expects the CMIP variables 'areacello', 'sivol', 'siconc' in a file structure ./Modelname/scenario/variable. The output of this algorithm is specified in the save_CMIP_results() function, and further reshaped and filtered in the utils.py. In the last step the data is plotted with the various plt_... functions.

Intermediate file output can be provided on request. 
