import sys
sys.path.insert(0,'../lib')
sys.path.insert(0,'../systems')
import numpy as np
import scipy.io as sio
from edgeAI_classes import *
from energy_management_system import *
import pdb

### parameter/options
code = 'c'
project_name = 'energy_management_system'
N = 12
max_iter = 2000
opt = edgeAI_Options(code,project_name,N,max_iter)

### system
sys = energy_management_system()

### problem
prob = None # for control with deep neural network

### neural networks
nn = edgeAI_NeuralNetwork("example")

### external data
data = sio.loadmat('external_data/wdata_processed.mat')
extdata = edgeAI_ExtData(data["extdata"])

### create edgeAI
edgy = edgeAI(opt,sys,prob,nn,extdata)

# choose which parts of the code should be generated
solver = 'nn' # either nn or qp
simulation = True
edgy.generate_code_mpc(solver,simulation)
