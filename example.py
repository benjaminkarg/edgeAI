import numpy as np
import scipy.io as sio
from lib.edgeAI_classes import *
from systems.room_temp_model import *
import pdb

### parameter/options
code = 'c'
project_name = 'example'
N = 12
max_iter = 2000
opt = edgeAI_Options(code,project_name,N,max_iter)

### system
sys = room_temp_model()

### problem
# prob = edgeAI_Problem(sys,opt)
prob = None

### neural networks
nn = edgeAI_NeuralNetwork("example")

### external data
data = sio.loadmat('data/wdata_processed.mat')
extdata = edgeAI_ExtData(data["extdata"])

### create edgeAI
edgy = edgeAI(opt,sys,prob,nn,extdata)

# choose which parts of the code should be generated
solver = 'nn' # either nn or qp
# simulation = True
# edgy.generate_code_mpc(solver,simulation)

# generate code for a dnn
device = 'fpga' # either "micro" for microcontroller or "fpga" for FPGA
edgy.generate_code_dnn(device)
