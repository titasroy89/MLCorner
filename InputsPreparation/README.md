The most popular packages for Machine Learning are python based. 

Data after (pre)selection with one of UHH2 models is stored in ROOT format and should be converted into numpy or some other format to be useful for python-based ML tools.

Pioneers of Deep Learning in our group (Andrea Malara and Arne Reimers) developed scripts for UHH2 ntuples to NumPy conversion, which became a base for UHH2toNumpy tool.
 
UHH2toNumpy reads branches from UHH2 AnalysisTree and stores them in numpy arrays.
The tool contains following pieces:
- steer.py to specify workflow, list of sample names, list of variables to be skipped in conversion 
- constants.py to specify paths to inputs and outputs, etc
- ModuleRunner.py with ReadoutMLVariables function, which sets variables for readout_to_numpy_arrays function from functions_ml.py 

As user you should modify only steer.py and constants.py, the rest should run smoothly:
1) cd UHH2toNumpy
2) modify  steer.py and constants.py
3) python steer.py


