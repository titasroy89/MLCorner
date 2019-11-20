# ZprimeNN
Collection of scripts for Z'->ttbar multiclasification NN including training and validation of both deterministic and Bayesian Neural Networks.

Tested with python 3.7 and tensorflow 2.0

## Bayesian NN
Run it on GPU node via:
source submit_steer_BNN.sh

"submit_steer_BNN.sh" sets user specific enviroment and calls "steer_BNN.py" script. 
"steer_BNN.py" prepares inputs and runs training and validation as well as plots various distributions for debugging and checking network performans.


## Deterministic NN
Run it on GPU node via:
source submit_steer_DNN.sh

The set-up follows the same structure as for BNN, but with deterministic network training and testing instead of BNN.