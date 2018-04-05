import numpy as np
import sys
sys.path.insert(0,'../lib')
from edgeAI_classes import edgeAI_System

def energy_management_system():
    """
        returns a struct containing all informations about a system
    """

    ### type of system
    typ = 'discrete'

    ### parameters
    alpha_d = 0.05
    alpha_w = 0.25
    n_d = 0.
    n_w = 0.

    ### system matrices
    A = np.atleast_2d([[0.8511-alpha_d*n_d, 0.0541, 0.0707, 0.0],
                       [0.1293, 0.8635-alpha_w*n_w, 0.0055, 0.0],
                            [0.0989, 0.0032, 0.7541, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

    B = np.atleast_2d([[0.0035, 0.0, 0.0, 0.0],
                       [0.0003, 0.0, 0.0, 0.0],
                       [0.0002, 0.0, 0.0, 0.0],
                       [0.0, -5.0, 0.0, 0.0]])

    E = 1e-3 * np.atleast_2d([[22.217, 1.7912, 42.2123],
                              [ 1.5376, 0.6944, 2.9214],
                              [103.1813, 0.1032, 196.0444],
                              [0.0, 0.0, 0.0]])

    F = []

    ### constraints
    Bx = np.vstack([np.diag(np.ones(3)),np.diag(-np.ones(3))])
    BN = np.vstack([np.diag(np.ones(3)),np.diag(-np.ones(3))])
    Bu = np.atleast_2d([[1.],[-1.]])

    dx = np.atleast_2d([[18.],[50],[50],[-16.],[50],[50]])
    dN = np.atleast_2d([[18.],[50],[50],[-16.],[50],[50]])
    du = np.atleast_2d([[1000.],[1000.]])

    ### objective
    Q = np.diag([1.,1.,1.])

    QN = np.diag([1.,1.,1.])

    R = np.diag([1.])

    ### create edgeAI_system
    system = edgeAI_System(A,B,E,F,Bx,BN,Bu,dx,dN,du,Q,QN,R,typ)

    return system
