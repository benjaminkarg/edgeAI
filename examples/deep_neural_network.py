import sys
sys.path.insert(0,'../lib')
sys.path.insert(0,'../systems')
import numpy as np
import scipy.io as sio
from edgeAI_classes import *
from room_temp_model import *

### parameter/options
code = 'c'
project_name = 'deep_neural_network'
opt = edgeAI_Options(code,project_name)

### neural networks
nn = edgeAI_NeuralNetwork(project_name)

### create edgeAI
edgy = edgeAI(opt,neuralnetwork=nn)

# generate code
device = 'fpga' # either "micro" for microcontroller or "fpga" for FPGA
edgy.generate_code_dnn(device)
