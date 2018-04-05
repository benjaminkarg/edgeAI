import sys
sys.path.insert(0,'../lib')
sys.path.insert(0,'../systems')
import numpy as np
import scipy.io as sio
from edgeAI_classes import *
from energy_management_system import *
import csv
import pdb

### parameter/options
code = 'c'
project_name = 'energy_management_system'
N = 24
opt = edgeAI_Options(code,project_name,N)

### system
sys = energy_management_system()

### neural networks
nn = edgeAI_NeuralNetwork('energy_management_system')

### external data
ext_vec = np.zeros([0,1])
with open('../external_data/weather_data.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        nv = np.reshape(float(row['weather_data']),(1,1))
        ext_vec = np.append(ext_vec,nv,axis=0)
extdata = edgeAI_ExtData(ext_vec)

### create edgeAI
edgy = edgeAI(opt=opt,sys=sys,neuralnetwork=nn,extdata=extdata)

# choose which parts of the code should be generated
solver = 'nn' # either nn or qp
simulation = True
edgy.generate_code_mpc(solver,simulation)
