import sys
sys.path.insert(0,'../lib')
sys.path.insert(0,'../systems')
import numpy as np
import scipy.io as sio
from edgeAI_classes import *
from room_temp_model import *
import csv
import pdb

### parameter/options
code = 'c'
project_name = 'energy_management_system'
N = 12
max_iter = 2000
opt = edgeAI_Options(code,project_name,N,max_iter)

### system
sys = room_temp_model()

### problem
# prob = edgeAI_Problem(sys,opt)

### neural networks
nn = None

### load external data
ext_vec = np.zeros([0,1])
with open('../external_data/weather_data.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        nv = np.reshape(float(row['weather_data']),(1,1))
        ext_vec = np.append(ext_vec,nv,axis=0)
extdata = edgeAI_ExtData(ext_vec)

### create edgeAI
edgy = edgeAI(opt,sys,prob,nn,extdata)

# choose which parts of the code should be generated
solver = 'qp'
simulation = True
edgy.generate_code_mpc(solver,simulation)
