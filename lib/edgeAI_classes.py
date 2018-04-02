import numpy as np
import json
import h5py
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import scipy.io as sio
from numpy.linalg import inv
import matlab.engine
import os
import pdb

class edgeAI_Options:
    """ A class containing all the options for code generation"""
    def __init__(self,code_type,project_name,horizon,max_iter):
        if (code_type == 'ino') or (code_type == 'c'):
            project_name += '_' + code_type
        else:
            raise Exception("Two options for code type. Choose either \"ino\" or \"c\"")
        self.code_type = code_type
        self.project_name = project_name
        self.N = horizon
        self.path = os.path.realpath(project_name)
        self.MAXITER = max_iter

class edgeAI_Matrix:
    """ A class saving matrices for edgeAI"""
    def __init__(self,mat,name,varname_size_rows=None,varname_size_cols=None):
        if mat == []:
            self.mat = np.array([])
        else:
            self.mat = mat
            self.mat[abs(self.mat)<1e-10] = 0
            self.rows, self.cols = mat.shape
            if varname_size_rows:
                self.vsr = varname_size_rows
            else:
                self.vsr = str(self.rows)
            if varname_size_cols:
                self.vsc = varname_size_cols
            else:
                self.vsc = str(self.cols)
            self.nnz = sum(sum(self.mat!=0))
            if self.nnz < (self.rows*(self.cols-1)-1)/2.:
                self.sparse = True
            else:
                self.sparse = False
            if self.sparse:
                self.mat_sparse = csr_matrix(self.mat)
            else:
                self.mat_sparse = []
        self.name = name

    def write(self,f):
        if (self.rows == 1 and self.cols == 1):
            f.write('const real_t '+self.name+' = ')
            f.write(str(self.mat[0,0]) +  ";\n")
        else:
            if self.sparse:
                f.write('const real_t '+self.name+'_data[] = {\n')
                for data in self.mat_sparse.data:
                    f.write(str(data)+', ')
                f.write('\n};\n')
                f.write('const uint32_t '+self.name+'_ptr[] = {\n')
                for ptr in self.mat_sparse.indptr:
                    f.write(str(ptr)+', ')
                f.write('\n};\n')
                f.write('const uint32_t '+self.name+'_ind[] = {\n')
                for ind in self.mat_sparse.indices:
                    f.write(str(ind)+', ')
                f.write('\n\n};\n')

            else:
                f.write('const real_t '+self.name+'[] = {\n')
                for row in range(self.rows):
                    for col in range(self.cols):
                        f.write(str(self.mat[row,col])+', ')
                    f.write('\n')
                f.write('\n};\n')

class edgeAI_ExtData:
    """
    A class containing external data
    """
    def __init__(self,mat):
        self.dist_vec = edgeAI_Matrix(mat,"dist_vec")

    def write_struct(self,f):
        f.write("\nstruct ext_data data = {dist_vec, in, d};")

class edgeAI_NeuralNetwork:
    """
    A class to extract the information
    from trained keras models (h5 and json file)
    """
    def __init__(self,net_name):

        ### read ~.json
        with open("nn_data/"+net_name+".json") as json_data:
            d_load = json.load(json_data)
            layers = d_load["config"]["layers"]
        self.act_fun = []
        for i in range(len(layers)-1):
            self.act_fun.append(layers[i+1]["config"]["activation"].encode())

        ### read ~.h5
        hf = h5py.File("nn_data/"+net_name+".h5", "r")
        layer_names = [n.decode('utf8') for n in hf.attrs['layer_names']]
        self.kernel = []
        self.bias = []
        for k, name in enumerate(layer_names):
            g = hf[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if weight_names:
                weight_values = [g[weight_name] for weight_name in weight_names]
                self.kernel.append(edgeAI_Matrix(weight_values[0][:].T,"kernel_"+str(k)))
                self.bias.append(edgeAI_Matrix(np.reshape(weight_values[1][:],(-1,1)),"bias_"+str(k)))
        self.num_layers = k

        ### read ~.mat file with scaling information
        scaling_data = sio.loadmat("nn_data/"+net_name+".mat")
        dif_in = scaling_data["up_in"] - scaling_data["low_in"]
        dif_inv_in = np.reshape(1./dif_in,(-1,1))
        self.dif_inv_in = edgeAI_Matrix(dif_inv_in,"dif_inv_in")
        low_in = np.reshape(scaling_data["low_in"],(-1,1))
        self.low_in = edgeAI_Matrix(low_in,"low_in")
        low_out = np.reshape(scaling_data["low_out"],(-1,1))
        self.low_out = edgeAI_Matrix(low_out,"low_out")
        dif_out = scaling_data["up_out"] - scaling_data["low_out"]
        dif_out = np.reshape(dif_out,(-1,1))
        self.dif_out = edgeAI_Matrix(dif_out,"dif_out")

    def write_data(self,f):
        self.dif_inv_in.write(f)
        self.low_in.write(f)
        self.low_out.write(f)
        self.dif_out.write(f)
        for k in range(self.num_layers):
            self.kernel[k].write(f)
            self.bias[k].write(f)

    def write_struct(self,f):
        f.write("\nstruct edgeAI_dnn dnn = {")
        f.write("dif_inv_in, low_in, low_out, dif_out, ")
        for k in range(self.num_layers):
            f.write("kernel_"+str(k+1)+",")
            if k < (self.num_layers-1):
                f.write(" bias_"+str(k+1)+", ")
            else:
                f.write(" bias_"+str(k+1)+"};\n")

    def write_arch(self,f):
        f.write("struct edgeAI_dnn{\n")
        f.write("const real_t *dif_inv_in;\n")
        f.write("const real_t *low_in;\n")
        f.write("const real_t *low_out;\n")
        f.write("const real_t *dif_out;\n")
        for k in range(self.num_layers):
            f.write("const real_t *kernel_"+str(k+1)+";\n")
            f.write("const real_t *bias_"+str(k+1)+";\n")
        f.write("};\n\n")

class edgeAI_System:
    """
    A class containing a dynamic (continuous or discrete)
        supported systems:
        x = Ax + Bu
        x = Ax + Bu + Ed
        x = Ax + Bu + Fz
        x = Ax + Bu + Ed + Fz
        x: state
        u: input
        d: disturbances
        z: couplings
    """
    def __init__(self,A,B,E=[],F=[],
                 Bx=[],BN=[],Bu=[],
                 dx=[],dN=[],du=[],
                 Q=[],QN=[],R=[],
                 typ=None):

        ### system matrices
        self.A = edgeAI_Matrix(A,"Ad","STATES","STATES")
        self.B = edgeAI_Matrix(B,"Bd","STATES","INPUTS")
        self.E = edgeAI_Matrix(E,"Ed","STATES","DISTURBANCES")
        self.F = edgeAI_Matrix(F,"Fd","STATES","COUPLINGS")

        ### constraints
        if Bx.size > 0:
            self.Bx = edgeAI_Matrix(Bx,"Bx","STATE_CONSTRAINTS","STATES")
            self.dx = edgeAI_Matrix(dx,"dx","STATE_CONSTRAINTS","1")
        else:
            self.Bx = np.atleast_2d([])
            self.dx = np.atleast_2d([])
        if BN.size > 0:
            self.BN = edgeAI_Matrix(BN,"BN","T_STATE_CONSTRAINTS","STATES")
            self.dN = edgeAI_Matrix(dx,"dN","T_STATE_CONSTRAINTS","1")
        else:
            self.BN = np.atleast_2d([])
            self.dN = np.atleast_2d([])
        if Bu.size > 0:
            self.Bu = edgeAI_Matrix(Bu,"Bu","INPUT_CONSTRAINTS","INPUTS")
            self.du = edgeAI_Matrix(du,"du","INPUT_CONSTRAINTS","1")
        else:
            self.Bu = np.atleast_2d([])
            self.du = np.atleast_2d([])

        ### objective
        self.Q = edgeAI_Matrix(Q,"Q","STATES","STATES")
        self.QN = edgeAI_Matrix(QN,"QN","STATES","STATES")
        self.R = edgeAI_Matrix(R,"R","INPUTS","INPUTS")

        ### sizes
        self.nx = A.shape[1]
        self.nu = B.shape[1]
        if E != []:
            self.nd = E.shape[1]
        else:
            self.nd = []
        if F != []:
            self.nz = F.shape[1]
        else:
            self.nz = []

        ### system type
        if typ == None:
            self.typ = 'discrete'
        elif (typ != 'discrete') and (typ != 'continuous'):
            raise Exception("\nChoose either \"discrete\" or \"continuous\"!")
        else:
            self.typ = typ

    def sim_step(self,x,u,d=None,z=None):
        if self.typ == 'discrete':
            if (self.E == None) and (d != None):
                raise Exception("System has no disturbance matrix E")
            if (self.F == None) and (z != None):
                raise Exception("System has no couplings matrix F")
            if self.E == None:
                if self.F == None:
                    x = np.dot(self.A.mat,x) + np.dot(self.B.mat,u)
                else:
                    x = np.dot(self.A.mat,x) + np.dot(self.B.mat,u) + \
                        np.dot(self.F.mat,z)
            elif self.F == None:
                x = np.dot(self.A.mat,x) + np.dot(self.B.mat,u) + \
                    np.dot(self.E.mat,d)
            else:
                x = np.dot(self.A.mat,x) + np.dot(self.B.mat,u) + \
                    np.dot(self.E.mat,d) + np.dot(self.F.mat,z)

        elif self.typ == 'continuous':
            ### TODO: add support for continous systems
            raise Exception("\nContinous systems are not yet supported!\n")

    def write_data(self,f):
        self.A.write(f)
        self.B.write(f)
        if self.E.mat.size > 0:
            self.E.write(f)
        if self.F.mat.size > 0:
            self.F.write(f)

    def write_struct(self,f):
        if self.A.sparse:
            sys_str = "Ad_data, Ad_ptr, Ad_ind"
        else:
            sys_str = "Ad"
        if self.B.sparse:
            sys_str += ", Bd_data, Bd_ptr, Bd_ind"
        else:
            sys_str += ", Bd"
        if self.E.mat.size > 0:
            if self.E.sparse:
                sys_str += ", Ed_data, Ed_ptr, Ed_ind"
            else:
                sys_str += ", Ed"
        if self.F.mat.size > 0:
            if self.F.sparse:
                sys_str += ", Fd_data, Fd_ptr, Fd_ind"
            else:
                sys_str = sys_str + ", Fd"
        f.write("\nstruct edgeAI_sys sys = {" + sys_str + "};")

    def write_arch(self,f):
        f.write("struct edgeAI_sys{\n")
        if self.A.sparse:
            f.write("real_t *Ad_data;\nuint32_t *Ad_ptr;\nuint32_t *Ad_ind;\n")
        else:
            f.write("real_t *Ad;\n")
        if self.B.sparse:
            f.write("real_t *Bd_data;\nuint32_t *Bd_ptr;\nuint32_t *Bd_ind;\n")
        else:
            f.write("real_t *Bd;\n")
        if self.E.mat.size > 0:
            if self.E.sparse:
                f.write("real_t Ed_data;\nuint32_t Ed_ptr;\nuint32_t Ed_ind;\n")
            else:
                f.write("real_t *Ed;\n")
        if self.F.mat.size > 0:
            if self.F.sparse:
                f.write("real_t *Fd_data;\nuint32_t *Fd_ptr;\nuint32_t *Fd_ind;\n")
            else:
                f.write("real_t *Fd;\n")
        f.write("};\n\n")

class edgeAI_Problem:
    """ A class containing the optimization problem """
    def __init__(self,sys,opt):

        ### rename
        N = opt.N

        ### initialize matrices
        E = []
        F = []
        Hq = []
        Hr = []
        H_d = []
        H_z = []

        A = np.zeros([(N+1)*sys.nx,(N+1)*sys.nx+N*sys.nu])
        A[0:sys.nx,0:sys.nx] = np.eye(sys.nx)
        b = np.zeros([(N+1)*sys.nx,sys.nx])
        b[0:sys.nx,:] = np.eye(sys.nx)

        for i in range(N):
            if i == 0:
                Bx = sys.Bx.mat
                Bu = sys.Bu.mat
                dx = sys.dx.mat
                du = sys.du.mat
                Hq = sys.Q.mat
                Hr = sys.R.mat
                if not sys.E == []:
                    E = sys.E.mat
                if not sys.F == []:
                    F = sys.F.mat
            else:
                Bx = block_diag(Bx,sys.Bx.mat)
                Bu = block_diag(Bu,sys.Bu.mat)
                dx = np.vstack([dx,sys.dx.mat])
                du = np.vstack([du,sys.du.mat])
                Hq = block_diag(Hq,sys.Q.mat)
                Hr = block_diag(Hr,sys.R.mat)
                if not sys.E == []:
                    E = block_diag(E,sys.E.mat)
                if not sys.F == []:
                    F = block_diag(F,sys.F.mat)
            ii = i+1
            iii = i+2
            A[ii*sys.nx:iii*sys.nx,i*sys.nx:iii*sys.nx] = np.hstack([sys.A.mat,-np.eye(sys.nx)])
            A[ii*sys.nx:iii*sys.nx,(N+1)*sys.nx+i*sys.nu:(N+1)*sys.nx+ii*sys.nu] = sys.B.mat

        B = block_diag(Bx,sys.BN.mat,Bu)
        d = np.vstack([dx,sys.dN.mat,du])
        d[d==-np.inf] = -1e10
        d[d==np.inf] = 1e10
        H = block_diag(Hq,sys.QN.mat,Hr)
        Hinv = inv(H)
        HA = np.dot(np.dot(A,Hinv),A.T)
        HAinv = inv(HA)
        AT = A.T

        H_b = np.dot(np.dot(Hinv,AT),HAinv)
        H_x = np.dot(H_b,b)
        H1_nue = np.dot(np.dot(np.dot(H_b,A),Hinv),B.T)
        H2_nue = np.dot(Hinv,B.T)
        H_nue = H1_nue - H2_nue
        if sys.E.mat.size > 0:
            E = np.vstack([np.zeros([sys.nx,E.shape[1]]),E])
            H_d = np.dot(H_b,E)
        if sys.F.mat.size > 0:
            F = np.vstack([np.zeros([sys.nx,F.shape[1]]),F])
            H_z = np.dot(H_b,F)

        ### TODO: Solution without matlab and yalmip
        bound = np.dot(np.dot(B,Hinv),B.T)
        sio.savemat('bound.mat',{'bound':bound})
        eng = matlab.engine.start_matlab()
        Lmat = eng.solve_for_L("bound.mat")
        L = sio.loadmat("L.mat")['L']
        Linv = inv(L)
        LinvB = np.dot(Linv,B)
        Linvd = np.dot(Linv,d)
        self.nm = B.shape[0]

        ### convert to edgeAI_Matrix
        self.H_x = edgeAI_Matrix(H_x,"H_x","OPT_VAR","STATES")
        self.H_nue = edgeAI_Matrix(H_nue,"H_nue","OPT_VAR","MULT")
        if not sys.E == []:
            self.H_d = edgeAI_Matrix(H_d,"H_d","OPT_VAR","HOR_DISTURBANCES")
        if not sys.F == []:
            self.H_z = edgeAI_Matrix(H_z,"H_z","OPT_VAR","HOR_COUPLINGS")
        self.LinvB = edgeAI_Matrix(LinvB,"LinvB","MULT","OPT_VAR")
        self.Linvd = edgeAI_Matrix(Linvd,"Linvd","MULT","1")

        ### clean directory
        os.remove("bound.mat")
        os.remove("L.mat")

    def write_data(self,f):
        self.H_x.write(f)
        self.H_nue.write(f)
        if self.H_d.mat.size > 0:
            self.H_d.write(f)
        if self.H_z.mat.size > 0:
            self.H_z.write(f)
        self.LinvB.write(f)
        self.Linvd.write(f)

    def write_struct(self,f):
        if self.H_x.sparse:
            prob_str = "H_x_data, H_x_ptr, H_x_ind"
        else:
            prob_str = "H_x"
        if self.H_nue.sparse:
            prob_str += ", H_nue_data, H_nue_ptr, H_nue_ind"
        else:
            prob_str += ", H_nue"
        if self.H_d.mat.size > 0:
            if self.H_d.sparse:
                prob_str += ", H_d_data, H_d_ptr, H_d_ind"
            else:
                prob_str += ", H_d"
        if self.H_z.mat.size > 0:
            if self.H_z.sparse:
                prob_str += ", H_z_data, H_z_ptr, H_z_ind"
            else:
                prob_str += ", H_z"
        if self.LinvB.sparse:
            prob_str += ", LinvB_data, LinvB_ptr, LinvB_ind"
        else:
            prob_str += ", LinvB"
        if self.Linvd.sparse:
            prob_str += ", Linvd_data, Linvd_ptr, Linvd_ind"
        else:
            prob_str += ", Linvd"
        f.write("\nstruct edgeAI_prob prob = {" + prob_str + ", tk};")

    def write_arch(self,f):
        f.write("struct edgeAI_prob{\n")
        if self.H_x.sparse:
            f.write("real_t *H_x_data;\nuint32_t *H_x_ptr;\nuint32_t *H_x_ind;\n")
        else:
            f.write("real_t *H_x;\n")
        if self.H_nue.sparse:
            f.write("real_t *H_nue_data;\nuint32_t *H_nue_ptr;\nuint32_t *H_nue_ind;\n")
        else:
            f.write("real_t *H_nue;\n")
        if self.H_d.mat.size > 0:
            if self.H_d.sparse:
                f.write("real_t *H_d_data;\nuint32_t *H_d_ptr;\nuint32_t *H_d_ind;\n")
            else:
                f.write("real_t *H_d;\n")
        if self.H_z.mat.size > 0:
            if self.H_z.sparse:
                f.write("real_t *H_z_data;\nuint32_t *H_z_ptr;\nuint32_t *H_z_ind;\n")
            else:
                f.write("real_t *H_z;\n")
        if self.LinvB.sparse:
            f.write("real_t *LinvB_data;\nuint32_t *LinvB_ptr;\nuint32_t *LinvB_ind;\n")
        else:
            f.write("real_t *LinvB;\n")
        if self.Linvd.sparse:
            f.write("real_t *Linvd_data;\nuint32_t *Linvd_ptr;\nuint32_t *Linvd_ind;\n")
        else:
            f.write("real_t *Linvd;\n")
        f.write("real_t *tk;\n")
        f.write("};\n\n")

class edgeAI:
    """ A class containing everything including chicken wings, guacamole and chapati """
    def __init__(self,opt,sys=None,prob=None,neuralnetwork=None,extdata=None):
        self.opt = opt
        self.sys = sys
        self.prob = prob
        self.nn = neuralnetwork
        self.ed = extdata

    def generate_code_mpc(self,solver,simulation):

        self.opt.gen_sim = True
        if (solver == "qp") or (solver == "nn"):
            self.opt.solver = solver
        else:
            raise Exception("\nChoose either \"qp\" or \"nn\"\n")

        if self.opt.code_type == "ino":
            if not os.path.isdir(self.opt.project_name):
                os.mkdir(self.opt.project_name)

        elif self.opt.code_type == "c":
            if not os.path.isdir(self.opt.project_name):
                os.mkdir(self.opt.project_name)
            if not os.path.isdir(self.opt.project_name+"/edgeAI"):
                os.mkdir(self.opt.project_name+"/edgeAI")
            if not os.path.isdir(self.opt.project_name+"/edgeAI/include"):
                os.mkdir(self.opt.project_name+"/edgeAI/include")

            self.write_makefiles()

        self.write_mtx_opts()

        self.write_mc04types()

        self.write_edgeAI_arch()

        self.write_edgeAI_main()

        self.write_stdint()

        self.write_edgeAI_const()

        self.write_edgeAI()

        print("The C-Code was saved in: \"" + self.opt.path + "\"")

    def generate_code_dnn(self,device):

        self.opt.solver = "nn"

        if self.opt.code_type == "ino":
            if not os.path.isdir(self.opt.project_name):
                os.mkdir(self.opt.project_name)

        elif self.opt.code_type == "c":
            if not os.path.isdir(self.opt.project_name):
                os.mkdir(self.opt.project_name)
            if not os.path.isdir(self.opt.project_name+"/edgeAI"):
                os.mkdir(self.opt.project_name+"/edgeAI")
            if not os.path.isdir(self.opt.project_name+"/edgeAI/include"):
                os.mkdir(self.opt.project_name+"/edgeAI/include")

            self.write_makefiles()

        self.write_mc04types()

        self.write_stdint()

        self.write_edgeAI_arch("nn")

        self.write_edgeAI_main("nn",device)

        self.write_edgeAI_const("nn")

        self.write_mtx_opts("nn",device)

        self.write_edgeAI("nn")

        print("The C-Code was saved in: \"" + self.opt.path + "\"")

    def write_mtx_opts(self,mode=None,device=None):

        if self.opt.code_type == 'c':
            path_h = self.opt.path + "/edgeAI/include/mtx_ops.h"
            path_c = self.opt.path + "/edgeAI/mtx_ops.c"
        elif self.opt.code_type == 'ino':
            path_h = self.opt.path + "/mtx_ops.h"
            path_c = self.opt.path + "/mtx_ops.c"

        ### write h-file
        with open(path_h,"w") as f:

            if mode == "nn":

                if device == "micro":

                    self.wdtc(f,"MTX_OPS_H")
                    self.witc(f,"edgeAI_arch.h")

                    self.wftc(f,"mtx_times_vec_dense",
                                ["pout","pmtx","pvec","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [     0,     1,     1,     1,     1],
                                [     1,     1,     1,     0,     0],
                                "e")

                    self.wftc(f,"mtx_times_vec_sparse",
                                ["pout","data","ptr","ind","vec","rows"],
                                ["real_t","real_t","uint32_t","uint32_t","real_t","uint32_t"],
                                [      0,      1,      1,      1,      1,       1],
                                [      1,      1,      1,      1,      1,       0],
                                "e")

                    self.wftc(f,"mult_scale",
                                ["out","in","sca","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [     0,     1,     1,     1,     1],
                                [     1,     1,     1,     0,     0],
                                "e")

                    self.wftc(f,"mtx_add",
                                ["pmtxc","pmtxa","pmtxb","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [     0,     1,     1,     1,     1],
                                [     1,     1,     1,     0,     0],
                                "e")

                    self.wftc(f,"mtx_substract",
                                ["pmtxc","pmtxa","pmtxb","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [     0,     1,     1,     1,     1],
                                [     1,     1,     1,     0,     0],
                                "e")

                    if "relu" in self.nn.act_fun:
                        self.wftc(f,"mtx_max_vec_zero",
                                    ["pmax","rows"],
                                    ["real_t","uint32_t"],
                                    [     0,     1],
                                    [     1,     0],
                                    "e")

                    if "tanh" in self.nn.act_fun:
                        self.wftc(f,"mtx_tanh",
                                 ["vec","rows"],
                                 ["real_t","uint32_t"],
                                 [       0,        1],
                                 [       1,        0],
                                 "e")

                    self.wdtc(f)

                elif device == "fpga":

                    self.wdtc(f,"MTX_OPS_H")
                    self.witc(f,"edgeAI_arch.h")
                    f.write("\n")

                    for layer in range(self.nn.num_layers):

                        self.fpga_layer(layer,"h",f)

                    self.fpga_scale("h",f)

                    self.fpga_unscale("h",f)

                    self.wdtc(f)

            else:

                self.wdtc(f,"MTX_OPS_H")
                self.witc(f,"edgeAI_arch.h")

                self.wftc(f,"mtx_times_vec_dense",
                            ["pout","pmtx","pvec","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [     0,     1,     1,     1,     1],
                            [     1,     1,     1,     0,     0],
                            "e")

                self.wftc(f,"mtx_times_vec_sparse",
                            ["pout","data","ptr","ind","vec","rows"],
                            ["real_t","real_t","uint32_t","uint32_t","real_t","uint32_t"],
                            [      0,      1,      1,      1,      1,       1],
                            [      1,      1,      1,      1,      1,       0],
                            "e")

                self.wftc(f,"mtx_scale",
                            ["pout","pmtx","factor","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [     0,     1,     1,     1,     1],
                            [     1,     1,     0,     0,     0],
                            "e")

                if self.opt.solver == "nn":
                    self.wftc(f,"mult_scale",
                                ["out","in","sca","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [     0,     1,     1,     1,     1],
                                [     1,     1,     1,     0,     0],
                                "e")

                self.wftc(f,"mtx_add",
                            ["pmtxc","pmtxa","pmtxb","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [     0,     1,     1,     1,     1],
                            [     1,     1,     1,     0,     0],
                            "e")

                self.wftc(f,"mtx_substract",
                            ["pmtxc","pmtxa","pmtxb","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [     0,     1,     1,     1,     1],
                            [     1,     1,     1,     0,     0],
                            "e")

                self.wftc(f,"mtx_saturate_vec",
                            ["pvec","plower","pupper","rows"],
                            ["real_t","real_t","real_t","uint32_t"],
                            [     0,     1,     1,     1],
                            [     1,     1,     1,     0],
                            "e")

                self.wftc(f,"mtx_max_vec_zero",
                            ["pmax","rows"],
                            ["real_t","uint32_t"],
                            [     0,     1],
                            [     1,     0],
                            "e")

                self.wftc(f,"mtx_min_vec_zero",
                            ["pmin","rows"],
                            ["real_t","uint32_t"],
                            [     0,     1],
                            [     1,     0],
                            "e")

                if self.opt.solver == "nn":
                    if "tanh" in self.nn.act_fun:
                        self.wftc(f,"mtx_tanh",
                                 ["vec","rows"],
                                 ["real_t","uint32_t"],
                                 [       0,        1],
                                 [       1,        0],
                                 "e")

                self.wdtc(f)

        ###write c-file
        with open(path_c,"w") as f:

            if mode == "nn":

                self.witc(f,"mtx_ops.h")
                if "tanh" in self.nn.act_fun:
                    self.wdtc(f,"MATH")
                    self.witc(f,"<math.h>")
                    self.wdtc(f)

                if device == "micro":

                    ## mtx_times_vec_dense
                    self.wftc_he(f,"mtx_times_vec_dense",
                                ["pout[]","pmtx[]","pvec[]","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [0,1,1,1,1],[1,1,1,0,0])

                    self.wftc_lv(f,["i","j","k = 0"],["uint32_t","uint32_t","uint32_t"])

                    # f.write("\t")
                    self.wfltc(f,0,"rows")
                    f.write("\t\tpout[i] = 0;\n\t\t")
                    self.wfltc(f,0,"cols",var="j")
                    f.write("\t\t\t pout[i] += pmtx[k] * pvec[j];\n\t\t\t\tk++;\n\t\t}\n\t}")

                    self.wftc_he(f)

                    ### mtx_times_vec_sparse
                    self.wftc_he(f,"mtx_times_vec_sparse",
                                 ["pout[]","data[]","ptr[]","ind[]","vec[]","rows"],
                                 ["real_t","real_t","uint32_t","uint32_t","real_t","uint32_t"],
                                 [0,1,1,1,1,1,1],[1,1,1,1,1,1,0,0])

                    self.wftc_lv(f,["i","j"],
                                   ["uint32_t","uint32_t"])

                    self.wfltc(f,0,"rows")
                    f.write("\t\tpout[i] = 0;\n\t\t")
                    self.wfltc(f,"ptr[i]","ptr[i+1]",var="j")
                    # f.write("\t\t\tr = ind[j];\n")
                    # f.write("\t\t\tk = ;\n")
                    f.write("\t\t\tpout[i] += data[j]*vec[ind[j]];\n\t\t}\n\t}")

                    self.wftc_he(f)

                    ### mult_scale
                    self.wftc_he(f,"mult_scale",
                                ["out[]","in[]","sca[]","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [0,1,1,1,1],[1,1,1,0,0])

                    self.wftc_lv(f,["k"],["uint32_t"])

                    self.wfltc(f,0,"rows * cols",var="k")
                    f.write("\tout[k] = in[k] * sca[k];\n\t}")

                    self.wftc_he(f)

                    ### mtx_add
                    self.wftc_he(f,"mtx_add",
                                ["pmtxc[]","pmtxa[]","pmtxb[]","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [0,1,1,1,1],[1,1,1,0,0])

                    self.wftc_lv(f,["k"],["uint32_t"])

                    self.wfltc(f,0,"rows * cols",var="k")
                    f.write("\t\tpmtxc[k] = pmtxa[k] + pmtxb[k];\n\t}")

                    self.wftc_he(f)

                    ### mtx_substract
                    self.wftc_he(f,"mtx_substract",
                                ["pmtxc[]","pmtxa[]","pmtxb[]","rows","cols"],
                                ["real_t","real_t","real_t","uint32_t","uint32_t"],
                                [0,1,1,1,1],[1,1,1,0,0])

                    self.wftc_lv(f,["k"],["uint32_t"])

                    self.wfltc(f,0,"rows * cols",var="k")
                    f.write("\t\tpmtxc[k] = pmtxa[k] - pmtxb[k];\n\t}")

                    self.wftc_he(f)

                    ### mtx_max_vec_zero
                    if "relu" in self.nn.act_fun:
                        self.wftc_he(f,"mtx_max_vec_zero",
                                    ["pmax[]","rows"],
                                    ["real_t","uint32_t"],
                                    [0,1],[1,0])

                        self.wftc_lv(f,["i","zero = 0."],["uint32_t","real_t"])
                        self.wfltc(f,0,"rows")
                        f.write("\t\tif (pmax[i] < zero) {\n\t\t\t")
                        f.write("pmax[i] = zero;\n\t\t}\n\t}\n}\n")

                    ### mtx_tanh
                    if "tanh" in self.nn.act_fun:
                        self.wftc_he(f,"mtx_tanh",
                                    ["vec[]","rows"],
                                    ["real_t","uint32_t"],
                                    [0,1],[1,0])
                        self.wftc_lv(f,["i"],["uint32_t"])
                        self.wfltc(f,0,"rows")
                        f.write("\t\tvec[i] = tanh(vec[i]);\n\t}\n}\n")

                elif device == "fpga":

                    for layer in range(self.nn.num_layers):
                        self.fpga_layer(layer,"c",f)
                    self.fpga_scale("c",f)
                    self.fpga_unscale("c",f)

            else:

                self.witc(f,"mtx_ops.h")
                if self.opt.solver == "nn":
                    if "tanh" in self.nn.act_fun:
                        self.wdtc(f,"MATH")
                        self.witc(f,"<math.h>")
                        self.wdtc(f)

                ## mtx_times_vec_dense
                self.wftc_he(f,"mtx_times_vec_dense",
                            ["pout[]","pmtx[]","pvec[]","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [0,1,1,1,1],[1,1,1,0,0])

                self.wftc_lv(f,["i","j","k = 0"],["uint32_t","uint32_t","uint32_t"])

                # f.write("\t")
                self.wfltc(f,0,"rows")
                f.write("\t\tpout[i] = 0;\n\t\t")
                self.wfltc(f,0,"cols",var="j")
                f.write("\t\t\t pout[i] += pmtx[k] * pvec[j];\n\t\t\t\tk++;\n\t\t}\n\t}")

                self.wftc_he(f)

                ### mtx_times_vec_sparse
                self.wftc_he(f,"mtx_times_vec_sparse",
                             ["pout[]","data[]","ptr[]","ind[]","vec[]","rows"],
                             ["real_t","real_t","uint32_t","uint32_t","real_t","uint32_t"],
                             [0,1,1,1,1,1,1],[1,1,1,1,1,1,0,0])

                self.wftc_lv(f,["i","j"],
                               ["uint32_t","uint32_t"])

                self.wfltc(f,0,"rows")
                f.write("\t\tpout[i] = 0;\n\t\t")
                self.wfltc(f,"ptr[i]","ptr[i+1]",var="j")
                # f.write("\t\t\tr = ind[j];\n")
                # f.write("\t\t\tk = ;\n")
                f.write("\t\t\tpout[i] += data[j]*vec[ind[j]];\n\t\t}\n\t}")

                self.wftc_he(f)

                ### mtx_scale
                self.wftc_he(f,"mtx_scale",
                            ["pout[]","pmtx[]","factor","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [0,1,1,1,1],[1,1,0,0,0])

                self.wftc_lv(f,["k"],["uint32_t"])

                self.wfltc(f,0,"rows * cols",var="k")
                f.write("\tpout[k] = pmtx[k] * factor;\n\t}")

                self.wftc_he(f)

                ### mult_scale
                self.wftc_he(f,"mult_scale",
                            ["out[]","in[]","sca[]","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [0,1,1,1,1],[1,1,1,0,0])

                self.wftc_lv(f,["k"],["uint32_t"])

                self.wfltc(f,0,"rows * cols",var="k")
                f.write("\tout[k] = in[k] * sca[k];\n\t}")

                self.wftc_he(f)

                ### mtx_add
                self.wftc_he(f,"mtx_add",
                            ["pmtxc[]","pmtxa[]","pmtxb[]","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [0,1,1,1,1],[1,1,1,0,0])

                self.wftc_lv(f,["k"],["uint32_t"])

                self.wfltc(f,0,"rows * cols",var="k")
                f.write("\t\tpmtxc[k] = pmtxa[k] + pmtxb[k];\n\t}")

                self.wftc_he(f)

                ### mtx_substract
                self.wftc_he(f,"mtx_substract",
                            ["pmtxc[]","pmtxa[]","pmtxb[]","rows","cols"],
                            ["real_t","real_t","real_t","uint32_t","uint32_t"],
                            [0,1,1,1,1],[1,1,1,0,0])

                self.wftc_lv(f,["k"],["uint32_t"])

                self.wfltc(f,0,"rows * cols",var="k")
                f.write("\t\tpmtxc[k] = pmtxa[k] - pmtxb[k];\n\t}")

                self.wftc_he(f)

                ### mtx_saturate_vec
                self.wftc_he(f,"mtx_saturate_vec",
                            ["pvec[]","plower[]","pupper[]","rows"],
                            ["real_t","real_t","real_t","uint32_t"],
                            [0,1,1,1],[1,1,1,0])

                self.wftc_lv(f,["i"],["uint32_t"])

                self.wfltc(f,0,"rows")
                f.write("\t\tif (pvec[i] > pupper[i]) {\n\t\t\t")
                f.write("pvec[i] = pupper[i];\n\t\t")
                f.write("} else if (pvec[i] < plower[i]) {\n\t\t\t")
                f.write("pvec[i] = plower[i];\n\t\t}\n\t}\n}\n")

                ### mtx_max_vec_zero
                self.wftc_he(f,"mtx_max_vec_zero",
                            ["pmax[]","rows"],
                            ["real_t","uint32_t"],
                            [0,1],[1,0])

                self.wftc_lv(f,["i","zero = 0."],["uint32_t","real_t"])
                self.wfltc(f,0,"rows")
                f.write("\t\tif (pmax[i] < zero) {\n\t\t\t")
                f.write("pmax[i] = zero;\n\t\t}\n\t}\n}\n")

                ### mtx_min_vec_zero
                self.wftc_he(f,"mtx_min_vec_zero",
                            ["pmin[]","rows"],
                            ["real_t","uint32_t"],
                            [0,1],[1,0])

                self.wftc_lv(f,["i","zero = 0."],["uint32_t","real_t"])
                self.wfltc(f,0,"rows")
                f.write("\t\tif (pmin[i] > zero) {\n\t\t\t")
                f.write("pmin[i] = zero;\n\t\t}\n\t}\n}\n")

                ### mtx_tanh
                if self.opt.solver == "nn":
                    if "tanh" in self.nn.act_fun:
                        self.wftc_he(f,"mtx_tanh",
                                    ["vec","rows"],
                                    ["real_t","uint32_t"],
                                    [0,1],[1,0])
                        self.wftc_lv(f,["i"],["uint32_t"])
                        self.wfltc(f,0,"rows")
                        f.write("\t\tvec[i] = tanh(vec[i]);\n\t}\n}\n")

    def write_mc04types(self):
        """
            A function to write the typedefs for number variables
        """

        if self.opt.code_type == 'c':
            path = self.opt.path + "/edgeAI/include/mc04types.h"
        elif self.opt.code_type == 'ino':
            path = self.opt.path + "/mc04types.h"

        with open(path,"w") as f:

            self.wdtc(f,"MC04TYPES_H")
            self.wdtc(f,"USE_MPC_STDINT")
            self.witc(f,"<stdint.h>")
            f.write("#else\n")
            self.witc(f,"edgeAI_stdint.h")
            self.wdtc(f)

            f.write("\ntypedef float float32_t;\n")
            f.write("typedef double float64_t;\n")
            f.write("typedef long double float128_t;\n")

            self.wdtc(f)

    def write_edgeAI_arch(self,mode=None):
        """
            A function to write the structs to ~.h
        """

        if self.opt.code_type == 'c':
            path = self.opt.path + "/edgeAI/include/edgeAI_arch.h"
        elif self.opt.code_type == 'ino':
            path = self.opt.path + "/edgeAI_arch.h"

        with open(path,"w") as f:

            if mode == "nn":

                self.wdtc(f,"EDGEAI_ARCH_H")

                self.witc(f,"mc04types.h")
                f.write("typedef float32_t real_t;\n\n")

                # initialize lists
                struct_list = []
                type_list = []
                ripe_list = []

                self.nn.write_arch(f)
                struct_list.append("edgeAI_dnn *dnn")
                type_list.append(3)
                ripe_list.append(1)

                struct_list += ["*in","*in_scaled","*out","*out_scaled"]
                type_list += [1,1,1,1]
                ripe_list += [0,0,0,0]

                self.wstc(f,"edgeAI_ctl",
                struct_list,
                type_list,
                ripe_list)

                self.wdtc(f)

            else:

                self.wdtc(f,"EDGEAI_ARCH_H")

                self.witc(f,"mc04types.h")
                f.write("typedef float32_t real_t;\n\n")

                # initialize lists
                struct_list = []
                type_list = []
                ripe_list = []

                if self.ed:
                    self.wstc(f,"ext_data",["*dist_vec","*in","*d"],[1,1,1],[1,0,0])
                    struct_list.append("ext_data *data")
                    type_list.append(3)
                    ripe_list.append(1)
                if self.opt.gen_sim:
                    self.sys.write_arch(f)
                    struct_list.append("edgeAI_sys *sys")
                    type_list.append(3)
                    ripe_list.append(1)
                if self.opt.solver == "qp":
                    self.prob.write_arch(f)
                    struct_list.append("edgeAI_prob *prob")
                    type_list.append(3)
                    ripe_list.append(1)
                if self.opt.solver == "nn":
                    self.nn.write_arch(f)
                    struct_list.append("edgeAI_dnn *dnn")
                    type_list.append(3)
                    ripe_list.append(1)
                struct_list += ["*x_trj","*u_trj","*z","*mue","*x","*u"]
                type_list += [1,1,1,1,1,1]
                ripe_list += [0,0,0,0,0,0]
                self.wstc(f,"edgeAI_ctl",
                struct_list,
                type_list,
                ripe_list)

                self.wdtc(f)

    def write_edgeAI_main(self,mode=None,device=None):
        """
            A function write
            Input:  path        absolute path to
        """

        if self.opt.code_type == 'c':
            path_h = self.opt.path + "/edgeAI/include/edgeAI_main.h"
            path_c = self.opt.path + "/edgeAI/edgeAI_main.c"
        elif self.opt.code_type == 'ino':
            path_h = self.opt.path + "/edgeAI_main.h"
            path_c = self.opt.path + "/edgeAI_main.c"

        ### write h-file
        with open(path_h,"w") as f:

            self.wdtc(f,"EDGEAI_MAIN_H")

            self.witc(f,"edgeAI_arch.h")
            self.witc(f,"mtx_ops.h")
            self.witc(f,"edgeAI_const.h")

            if mode == "nn":

                self.wftc(f,"make_dnn_step",
                         ["*ctl"],
                         ["struct edgeAI_ctl"],
                         [0],[0],"e")

            else:

                if self.ed:
                    self.wftc(f,"update_data",
                            ["*ctl","k"],
                            ["struct edgeAI_ctl","uint32_t"],
                            [       0,         1],
                            [       0,         0],
                            "e")

                self.wftc(f,"make_opt_step",
                        ["*ctl"],
                        ["struct edgeAI_ctl"],
                        [       0],
                        [       0],
                        "e")

                if self.opt.gen_sim:
                    self.wftc(f,"make_sim_step",
                        ["*ctl"],
                        ["struct edgeAI_ctl"],
                        [       0],
                        [       0],
                        "e")

            self.wdtc(f)

        ### write c-file
        with open(path_c,"w") as f:

            self.witc(f,"<stddef.h>")
            self.witc(f,"<string.h>")
            self.witc(f,"edgeAI_main.h")
            self.witc(f,"edgeAI_const.h")

            if self.opt.solver == "nn":
                self.wftc(f,"make_scale",
                        ["*ctl"],
                        ["struct edgeAI_ctl"],
                        [0], [0], "s")

                self.wftc(f,"make_unscale",
                        ["*ctl"],
                        ["struct edgeAI_ctl"],
                        [0], [0], "s")

            # wftc(f,"warmstart",["*ctl"],["struct edgeAI_ctl"],[0],[0],"s")

            if mode == "nn":

                ### run_dnn
                self.wftc_he(f,"run_dnn",
                          ["*ctl"],
                          ["struct edgeAI_ctl"],
                          [0],[0])

                var_list = []
                type_list = []
                for k in range(self.nn.num_layers):
                    var_list.append("x_layer_"+str(k+1)+"["+str(self.nn.kernel[k].rows)+"]")
                    type_list.append("real_t")
                self.wftc_lv(f,var_list,type_list)

                f.write("make_scale(ctl);\n\n")

                for k in range(self.nn.num_layers):
                    if device == "micro":
                        if k == 0:
                            f.write("\t")
                            self.mult(self.nn.kernel[k],"ctl->dnn->","ctl->in_scaled","x_layer_"+str(k+1),f)
                            f.write("\tmtx_add(x_layer_"+str(k+1)+",x_layer_"+str(k+1)+",ctl->dnn->bias_"+str(k+1)+","+str(self.nn.kernel[k].rows)+",1);\n")
                            if self.nn.act_fun[k] == "relu":
                                f.write("\tmtx_max_vec_zero(x_layer_"+str(k+1)+","+str(self.nn.kernel[k].rows)+");\n\n")
                            elif self.nn.act_fun[k] == "tanh":
                                f.write("\tmtx_tanh(x_layer_"+str(k+1)+","+str(self.nn.kernel[k].rows)+");\n\n")
                        elif k == int(self.nn.num_layers-1):
                            f.write("\t")
                            self.mult(self.nn.kernel[k],"ctl->dnn->","x_layer_"+str(k),"x_layer_"+str(k+1),f)
                            f.write("\tmtx_add(ctl->out_scaled,x_layer_"+str(k+1)+",ctl->dnn->bias_"+str(k+1)+","+str(self.nn.kernel[k].rows)+",1);\n")
                            if self.nn.act_fun[k] == "relu":
                                f.write("\tmtx_max_vec_zero(ctl->out_scaled,"+str(self.nn.kernel[k].rows)+");\n\n")
                            elif self.nn.act_fun[k] == "tanh":
                                f.write("\tmtx_tanh(ctl->out_scaled,"+str(self.nn.kernel[k].rows)+");\n\n")
                        else:
                            f.write("\t")
                            self.mult(self.nn.kernel[k],"ctl->dnn->","x_layer_"+str(k),"x_layer_"+str(k+1),f)
                            f.write("\tmtx_add(x_layer_"+str(k+1)+",x_layer_"+str(k+1)+",ctl->dnn->bias_"+str(k+1)+","+str(self.nn.kernel[k].rows)+",1);\n")
                            if self.nn.act_fun[k] == "relu":
                                f.write("\tmtx_max_vec_zero(x_layer_"+str(k+1)+","+str(self.nn.kernel[k].rows)+");\n\n")
                            elif self.nn.act_fun[k] == "tanh":
                                f.write("\tmtx_tanh(x_layer_"+str(k+1)+","+str(self.nn.kernel[k].rows)+");\n\n")
                    elif device == "fpga":
                        if k == 0:
                            f.write("\tmtx_times_vec_dense_%i(x_layer_%i,ctl->dnn->kernel_%i,ctl->in_scaled);\n" % (k+1,k+1,k+1))
                            f.write("\tmtx_add_"+str(k+1)+"(x_layer_"+str(k+1)+",x_layer_"+str(k+1)+",ctl->dnn->bias_"+str(k+1)+");\n")
                            if self.nn.act_fun[k] == "relu":
                                f.write("\tmtx_relu_"+str(k+1)+"(x_layer_"+str(k+1)+");\n\n")
                            elif self.nn.act_fun[k] == "tanh":
                                f.write("\tmtx_tanh_"+str(k+1)+"(x_layer_"+str(k+1)+");\n\n")
                        elif k == int(self.nn.num_layers-1):
                            f.write("\tmtx_times_vec_dense_%i(x_layer_%i,ctl->dnn->kernel_%i,x_layer_%i);\n" % (k+1,k+1,k+1,k))
                            f.write("\tmtx_add_"+str(k+1)+"(ctl->out_scaled,x_layer_"+str(k+1)+",ctl->dnn->bias_"+str(k+1)+");\n")
                            if self.nn.act_fun[k] == "relu":
                                f.write("\tmtx_relu_"+str(k+1)+"(ctl->out_scaled);\n\n")
                            elif self.nn.act_fun[k] == "tanh":
                                f.write("\tmtx_tanh_"+str(k+1)+"(ctl->out_scaled);\n\n")
                        else:
                            f.write("\tmtx_times_vec_dense_%i(x_layer_%i,ctl->dnn->kernel_%i,x_layer_%i);\n" % (k+1,k+1,k+1,k))
                            f.write("\tmtx_add_"+str(k+1)+"(x_layer_"+str(k+1)+",x_layer_"+str(k+1)+",ctl->dnn->bias_"+str(k+1)+");\n")
                            if self.nn.act_fun[k] == "relu":
                                f.write("\tmtx_relu_"+str(k+1)+"(x_layer_"+str(k+1)+");\n\n")
                            elif self.nn.act_fun[k] == "tanh":
                                f.write("\tmtx_tanh_"+str(k+1)+"(x_layer_"+str(k+1)+");\n\n")

                f.write("\tmake_unscale(ctl);\n")

                self.wftc_he(f)

                self.wftc_he(f,"make_scale",
                          ["*ctl"],
                          ["struct edgeAI_ctl"],
                          [0],[0],"s")

                self.wftc_lv(f,["in_dif[INPUT]"],["real_t"])
                if device == "micro":
                    f.write("mtx_substract(in_dif, ctl->in, ctl->dnn->low_in, "+str(self.nn.low_in.rows)+", 1);\n")
                    f.write("\tmult_scale(ctl->in_scaled, in_dif, ctl->dnn->dif_inv_in, "+str(self.nn.low_in.rows)+", 1);\n")
                elif device == "fpga":
                    f.write("mtx_substract_scale(in_dif, ctl->in, ctl->dnn->low_in);\n")
                    f.write("\tmult_scale(ctl->in_scaled, in_dif, ctl->dnn->dif_inv_in);\n")

                self.wftc_he(f)

                self.wftc_he(f,"make_unscale",
                          ["*ctl"],
                          ["struct edgeAI_ctl"],
                          [0],[0],"s")

                self.wftc_lv(f,["out_scaled_biased[OUTPUT]"],["real_t"])
                if device == "micro":
                    f.write("mult_scale(out_scaled_biased, ctl->out_scaled, ctl->dnn->dif_out, "+str(self.nn.low_out.rows)+", 1);\n")
                    f.write("\tmtx_add(ctl->out, out_scaled_biased, ctl->dnn->low_out, "+str(self.nn.low_out.rows)+", 1);\n")
                elif device == "fpga":
                    f.write("mult_unscale(out_scaled_biased, ctl->out_scaled, ctl->dnn->dif_out);\n")
                    f.write("\tmtx_add_unscale(ctl->out, out_scaled_biased, ctl->dnn->low_out);\n")

                self.wftc_he(f)

            else:

                ### update_data
                if self.ed:
                    self.wftc_he(f,"update_data",
                                ["*ctl","k"],
                                ["struct edgeAI_ctl","uint32_t"],
                                [0,1],[0,0])

                    self.wftc_lv(f,["i","j","m = 0"],["uint32_t","uint32_t","uint32_t"])

                    if self.nn:
                        self.wfltc(f,0,"STATES")
                        f.write("\t\tctl->data->in[i] = ctl->x[i];\n\t}\n\n\t")

                    self.wfltc(f,0,"HOR")
                    f.write("\t\t")
                    self.wfltc(f,0,"DISTURBANCES",var="j")
                    f.write("\t\t\t")
                    f.write("ctl->data->d[m] = ctl->data->dist_vec[(k+i)*DISTURBANCES+j];\n\t\t\t")
                    f.write("m++;\n")
                    f.write("\t\t}\n\t}")

                    self.wftc_he(f)

                ### make_opt_step
                self.wftc_he(f,"make_opt_step",
                          ["*ctl"],
                          ["struct edgeAI_ctl"],
                          [0],[0])

                if self.opt.solver == "qp":

                    var_list = ["i","k","H_x[OPT_VAR]",
                               "yk[OPT_VAR]","LBy[MULT]","dif_mue[MULT]","mue_new[MULT]",
                               "mue_old[MULT]","nue[MULT]"]
                    type_list = ["uint32_t","uint32_t","real_t","real_t","real_t",
                                 "real_t","real_t","real_t","real_t"]
                    if self.sys.E.mat.size > 0:
                        var_list.append("H_d[MULT]")
                        type_list.append("real_t")
                    if self.sys.F.mat.size > 0:
                        var_list.append("H_z[MULT]")
                        type_list.append("real_t")
                    self.wftc_lv(f,var_list,type_list)

                    ### precomputations for 31
                    if self.sys.E.mat.size > 0:
                        self.mult(self.prob.H_d,"ctl->prob->","ctl->data->d","H_d",f);
                    if self.sys.F.mat.size > 0:
                        self.mult(self.prob.H_d,"ctl->prob->","ctl->data->d","H_d",f);
                    f.write("\t")
                    self.mult(self.prob.H_x,"ctl->prob->","ctl->x","H_x",f);
                    f.write("\n\t")
                    self.wfltc(f,0,"MULT",var="k")
                    f.write("\t\tmue_old[k] = 0.;\n")
                    f.write("\t\tnue[k] = 0.;\n\t}\n")

                    f.write("\n\t")
                    self.wfltc(f,0,"MAXITER")
                    f.write("\n\t\t")

                    ### 31
                    self.mult(self.prob.H_nue,"ctl->prob->","nue","yk",f);
                    # f.write("\t\tmtx_times_vec_dense(yk,ctl->prob->H_nue,nue,OPT_VAR,MULT);\n")
                    if self.sys.E.mat.size > 0:
                        f.write("\t\tmtx_substract(yk,yk,H_d,OPT_VAR,1);\n")
                    if self.sys.F.mat.size > 0:
                        f.write("\t\tmtx_substract(yk,yk,H_z,OPT_VAR,1);\n")
                    f.write("\t\tmtx_add(yk,yk,H_x,OPT_VAR,1);\n\n\t\t")

                    ### 32
                    self.mult(self.prob.LinvB,"ctl->prob->","yk","LBy",f);
                    # f.write("\t\tmtx_times_vec_dense(LBy,ctl->prob->LinvB,yk,MULT,OPT_VAR);\n")
                    f.write("\t\tmtx_substract(dif_mue,LBy,ctl->prob->Linvd,MULT,1);\n")
                    f.write("\t\tmtx_add(mue_new,nue,dif_mue,MULT,1);\n")
                    f.write("\t\tmtx_max_vec_zero(mue_new,MULT);\n\n")

                    ### 34 (33 precomputed)
                    f.write("\t\tmtx_substract(nue,mue_new,mue_old,MULT,1);\n")
                    f.write("\t\tmtx_scale(nue,nue,ctl->prob->tk[i],MULT,1);\n")
                    f.write("\t\tmtx_add(nue,mue_new,nue,MULT,1);\n\t\t")
                    self.wfltc(f,0,"MULT",var="k")
                    f.write("\t\t\tmue_old[k] = mue_new[k];\n\t\t}\n")
                    f.write("\n\t}\n\n\t")

                    ### copy variables to struct
                    self.wfltc(f,0,"STATES+HOR_STATES",var="k")
                    f.write("\t\tctl->x_trj[k] = yk[k];\n\t}\n\t")
                    self.wfltc(f,0,"HOR_INPUTS",var="k")
                    f.write("\t\tctl->u_trj[k] = yk[STATES+HOR_STATES+k];\n\t}\n\t")
                    self.wfltc(f,0,"OPT_VAR",var="k")
                    f.write("\t\tctl->z[k] = yk[k];\n\t}\n\t")
                    self.wfltc(f,0,"MULT",var="k")
                    f.write("\t\tctl->mue[k] = mue_new[k];\n\t}\n\t")
                    self.wfltc(f,0,"INPUTS",var="k")
                    f.write("\t\tctl->u[k] = ctl->u_trj[k];\n\t}\n\t")

                elif self.opt.solver == "nn":

                    var_list = ["k"]
                    type_list = ["uint32_t"]
                    for k in range(self.nn.num_layers-1):
                        var_list.append("x_layer_"+str(k+1)+"["+str(self.nn.kernel[k].rows)+"]")
                        type_list.append("real_t")
                    self.wftc_lv(f,var_list,type_list)

                    f.write("make_scale(ctl);\n\n")

                    for k in range(self.nn.num_layers):
                        if k == 0:
                            f.write("\t")
                            self.mult(self.nn.kernel[k],"ctl->dnn->","ctl->x","x_mean_"+str(k+1),f)
                            f.write("\tmtx_add(x_layer_"+str(k+1)+",x_layer_"+str(k+1)+",ctl->dnn->bias_"+str(k+1)+","+str(self.nn.kernel[k].rows)+",1);\n")
                            if self.nn.act_fun[k] == "relu":
                                f.write("\tmtx_max_vec_zero(x_layer_"+str(k+1)+","+str(self.nn.kernel[k].rows)+");\n\n")
                            elif self.nn.act_fun[k] == "tanh":
                                f.write("\tmtx_tanh(x_layer_"+str(k+1)+","+str(self.nn.kernel[k].rows)+");\n\n")
                        else:
                            f.write("\t")
                            self.mult(self.nn.kernel[k],"ctl->dnn->","x_mean_"+str(k),"x_mean_"+str(k+1),f)
                            f.write("\tmtx_add(x_layer_"+str(k+1)+",x_layer_"+str(k+1)+",ctl->dnn->bias_"+str(k+1)+","+str(self.nn.kernel[k].rows)+",1)\n")
                            if self.nn.act_fun[k] == "relu":
                                f.write("\tmtx_max_vec_zero(x_layer_"+str(k+1)+","+str(self.nn.kernel[k].rows)+");\n\n")
                            elif self.nn.act_fun[k] == "tanh":
                                f.write("\tmtx_tanh(x_layer_"+str(k+1)+","+str(self.nn.kernel[k].rows)+")\n\n")

                    f.write("\tmake_unscale(ctl);\n")

                self.wftc_he(f)

                if self.opt.gen_sim:
                    ### make_sime_step
                    self.wftc_he(f,"make_sim_step",
                              ["*ctl"],
                              ["struct edgeAI_ctl"],
                              [0],[0])

                    mean_list = ["Adx[STATES]","Bdu[STATES]"]
                    type_list = ["real_t","real_t"]
                    if self.sys.E.mat.size > 0:
                        mean_list.append("Edd[STATES]")
                        type_list.append("real_t")
                    if self.sys.F.mat.size > 0:
                        mean_list.append("Fdz[STATES]")
                        type_list.append("real_t")
                    self.wftc_lv(f,mean_list,type_list)

                    self.mult(self.sys.A,"ctl->sys->","ctl->x","Adx",f)
                    f.write("\t")
                    self.mult(self.sys.B,"ctl->sys->","ctl->u","Bdu",f)
                    if self.sys.E.mat.size > 0:
                        f.write("\t")
                        self.mult(self.sys.E,"ctl->sys->","ctl->data->d","Edd",f);
                    if self.sys.F.mat.size > 0:
                        f.write("\t")
                        self.mult(self.sys.F,"ctl->sys->","z","Fdz",f);

                    f.write("\n\tmtx_add(ctl->x, Adx, Bdu, STATES, 1);\n\t")
                    if self.sys.E.mat.size > 0:
                        f.write("mtx_add(ctl->x, ctl->x, Edd, STATES, 1);\n\t")
                    if self.sys.F.mat.size > 0:
                        f.write("mtx_add(ctl->x, ctl->x, Fdz, STATES, 1);\n\t")

                    self.wftc_he(f)

                if self.opt.solver == "nn":
                    self.wftc_he(f,"make_scale",
                              ["*ctl"],
                              ["struct edgeAI_ctl"],
                              [0],[0],"s")

                    self.wftc_lv(f,["in_dif["+str(self.nn.dif_inv_in.rows)+"]"],["real_t"])
                    f.write("mtx_substract(in_dif, ctl->data->in, ctl->dnn->low_in, "+str(self.nn.low_in.rows)+", 1);\n")
                    f.write("\tmult_scale(ctl->in_scaled, in_dif, ctl->dnn->dif_inv_in, "+str(self.nn.low_in.rows)+", 1);\n")

                    self.wftc_he(f)

                    self.wftc_he(f,"make_unscale",
                              ["*ctl"],
                              ["struct edgeAI_ctl"],
                              [0],[0],"s")

                    self.wftc_lv(f,["out_sca["+str(self.nn.dif_out.rows)+"]"],["real_t"])
                    f.write("mult_scale(out_sca, ctl->u_scaled, ctl->dnn->dif_out, "+str(self.nn.low_out.rows)+", 1);\n")
                    f.write("\tmtx_add(ctl->u, out_sca, ctl->dnn->low_out, "+str(self.nn.low_out.rows)+", 1);\n")

                    self.wftc_he(f)

                # wftc_he(f,"warmstart",["*ctl"],["struct edgeAI_ctl"],[0],[0],"s")
                #
                # wftc_lv(f,["i"],["uint32_t"])
                #
                # wfltc(f,0,"HOR_STATES")
                # f.write("\t\tctl->z[i] = ctl->z[i+STATES];\n\t}\n\n\t")
                #
                # wfltc(f,0,"HOR_INPUTS-INPUTS")
                # f.write("\t\tctl->z[i+STATES+HOR_STATES] = ctl->z[i+STATES+HOR_STATES+INPUTS];\n\t}\n\n\t")
                #
                # wfltc(f,0,"EQ_CON-STATES")
                # f.write("\t\tctl->lambda[i] = ctl->lambda[i+STATES];\n\t}")
                #
                # wftc_he(f)

    def write_stdint(self):
        """
            A function to write stdint_h-file
        """

        if self.opt.code_type == 'c':
            path = self.opt.path + "/edgeAI/include/edgeAI_stdint.h"
        elif self.opt.code_type == 'ino':
            path = self.opt.path + "/edgeAI_stdint.h"

        with open(path,"w") as f:

            self.wdtc(f,"STDINT_H")

            f.write("typedef char char_t;\n\n")

            f.write("# ifndef __int8_t_defined\n")
            f.write("#  define __int8_t_defined\n")
            f.write("typedef signed char int8_t;\n")
            f.write("typedef unsigned char uint8_t\n")
            self.wdtc(f)

            f.write("# ifndef __int16_t_defined\n")
            f.write("#  define __int16_t_defined\n")
            f.write("typedef signed short int16_t;\n")
            f.write("typedef unsigned short uint16_t\n")
            self.wdtc(f)

            f.write("# ifndef __int32_t_defined\n")
            f.write("#  define __int32_t_defined\n")
            f.write("typedef signed int int32_t;\n")
            f.write("typedef unsigned int uint32_t\n")
            self.wdtc(f)
            f.write("\n")

            f.write("#ifndef _SYS_TYPES_H\n")
            f.write("#if !defined(_INT64_T) && !defined(INT64_MAX)\n")
            f.write("#define _INT64_T\n")
            f.write("typedef signed long int64_t;\n")
            self.wdtc(f)
            self.wdtc(f)
            f.write("\n")

            f.write("#ifndef _SYS_TYPES_H\n")
            f.write("#if !defined(_UINT64_T) && !defined(UINT64_MAX)\n")
            f.write("#define _UINT64_T\n")
            f.write("typedef unsigned long uint64_t;\n")
            self.wdtc(f)
            self.wdtc(f)

    def write_edgeAI(self,mode=None):
        """
            bliablablubb
        """
        if self.opt.code_type == 'ino':

            with open(self.opt.path+"/"+self.opt.path+".ino","w") as f:

                if mode == "nn":

                    ### TODO: Implement ~.ino file
                    f.write("TODO: Implement")

                else:

                    f.write("extern \"C\" {\n#include \"edgeAI_main.h\"\n}\n\n")
                    f.write("extern struct edgeAI_ctl ctl;\n")
                    f.write("uint32_t k;\n\n")

                    ### setup
                    f.write("void setup() {\n\t")
                    f.write("Serial.begin(9600);\n\t") # initialize communication
                    f.write("while (!Serial);\n\n\t") # wait until Serial established

                    f.write("const float32_t ")
                    for i in range(self.sys.nx):
                        if i == self.sys.nx-1:
                            f.write("X"+str(i)+" = 0.;")
                        else:
                            f.write("X"+str(i)+(" = 0., "))
                    for i in range(self.sys.nx):
                        f.write("\n\tctl.x["+str(i)+"] = X"+str(i)+";")

                    f.write("\n\n\tk = 0;\n")

                    # step
                    f.write("\n\tSerial.print(\"iter\");\n\t")
                    # headline states
                    for i in range(self.sys.nx):
                        f.write("Serial.print(\"\\t\\tx["+str(i)+"],\");\n\t")
                    # headline inputs
                    for i in range(self.sys.nu):
                        if i == self.sys.nu-1:
                            f.write("Serial.println(\"\\t\\tu["+str(i)+"],\");\n\n}\n\n")
                        else:
                            f.write("Serial.print(\"\\t\\tu["+str(i)+"],\");\n\n}\n\n")

                    ### loop
                    f.write("void loop() {\n\n\t")
                    if self.ed:
                        f.write("update_data(&ctl, k);\n\n\t")
                        f.write("make_opt_step(&ctl);\n\n\t")
                    else:
                        f.write("make_opt_step(&ctl);\n\n\t")
                    if self.opt.gen_sim:
                        f.write("make_sim_step(&ctl);\n\n\t")

                    # print results
                    f.write("Serial.print(k);\n\t")
                    f.write("Serial.print(\",\\t\\t\");")

                    for i in range(self.sys.nx):
                        f.write("\n\tSerial.print(ctl.x["+str(i)+"],2);")
                        f.write("\n\tSerial.print(\",\\t\\t\");")

                    for i in range(self.sys.nu):
                        if i == self.sys.nu-1:
                            f.write("\n\tSerial.println(ctl.u["+str(i)+"],2);\n")
                        else:
                            f.write("\n\tSerial.print(ctl.u["+str(i)+"],2);")
                            f.write("\n\tSerial.print(\",\\t\\t\");\n\t")

                    f.write("\n\tk++;\n\n\t")

                    # to avoid endless loop/errors
                    if True:
                        f.write("while (k == 48) {\n\t\tdelay(10000);\n\t}")

                    f.write("\n\n}")

        elif self.opt.code_type == 'c':

            project_name = self.opt.project_name[0:-2]

            with open(self.opt.path+"/"+project_name+".c","w") as f:

                self.witc(f,"<stdio.h>")
                self.witc(f,"edgeAI/include/edgeAI_main.h")
                f.write("int main(void)\n{\n\t")
                f.write("int k;\n\t")
                f.write("extern struct edgeAI_ctl ctl;\n\t")

                if mode == "nn":

                    for i in range(self.nn.low_in.rows):
                        f.write("ctl.in["+str(i)+"] = 0.;\n\t")

                    for i in range(self.nn.low_out.rows):
                        if i == 0:
                            f.write("\n\tprintf(\"out["+str(i)+"]")
                        elif i == (self.nn.low_out.rows-1):
                            f.write("\\t\\tout["+str(i)+"]")
                        else:
                            f.write("\\t\\tout["+str(i)+"]")
                    f.write("\\n\");\n\n\t")

                    f.write("run_dnn(&ctl);\n\n\t")

                    self.wfltc(f,0,"OUTPUT",var="k")
                    f.write("\t\t\tprintf(\"\\t%f\",(ctl.out[k]));\n\t}\n\t")
                    f.write("printf(\"\\n\");\n")
                    f.write("\n\n\treturn 0;\n}")

                else:

                    for i in range(self.sys.nx):
                        f.write("ctl.x["+str(i)+"] = 0.;\n\t")
                    for i in range(self.sys.nu):
                        f.write("ctl.u["+str(i)+"] = 0.;\n\t")

                    self.wfltc(f,0,"MULT",var="k")
                    f.write("\t\tctl.mue[k] = 0.;\n\t}")

                    f.write("\n\tprintf(\"iter")
                    for i in range(self.sys.nx):
                        f.write("\\t\\tx["+str(i)+"]")
                    for i in range(self.sys.nu):
                        f.write("\\t\\tu["+str(i)+"]")
                    f.write("\\n\");\n\n\t")

                    self.wfltc(f,0,48)

                    # optimization and simulation
                    if self.ed:
                        f.write("\n\t\tupdate_data(&ctl,i);\n\n\t\t")
                        f.write("make_opt_step(&ctl);\n\n\t\t")
                    else:
                        f.write("\n\t\tmake_opt_step(&ctl);\n\n\t\t")
                    if self.sys:
                        f.write("make_sim_step(&ctl);\n\n\t\t")

                    # output (results)
                    f.write("printf(\"%d\\t\",i);\n\t\t")
                    self.wfltc(f,0,"STATES",var="k")
                    f.write("\t\t\tprintf(\"\\t%f\",(ctl.x[k]));\n\t\t}\n\t\t")
                    self.wfltc(f,0,"INPUTS",var="k")
                    f.write("\t\t\tprintf(\"\\t%f\",(ctl.u[k]));\n\t\t}\n\t\t")
                    f.write("printf(\"\\n\");\n\n\t}\n\n\treturn 0;\n}")

    def write_edgeAI_const(self,mode=None):
        """
            A function writing the constant values to a file
        """

        if self.opt.code_type == 'c':
            path_h = self.opt.path + "/edgeAI/include/edgeAI_const.h"
            path_c = self.opt.path + "/edgeAI/edgeAI_const.c"
        elif self.opt.code_type == 'ino':
            path_h = self.opt.path + "/edgeAI_const.h"
            path_c = self.opt.path + "/edgeAI_const.c"

        ### write h-file
        with open(path_h,"w") as f:

            if mode == "nn":

                self.wdtc(f,"EDGEAI_CONST_H")
                self.witc(f,"edgeAI_arch.h")

                f.write("\nenum {\n")
                f.write("INPUT = " + str(self.nn.low_in.rows) + ",\n")
                f.write("OUTPUT = " + str(self.nn.low_out.rows) + "\n};\n")

                self.wdtc(f)

            else:

                N = self.opt.N
                self.wdtc(f,"EDGEAI_CONST_H")
                self.witc(f,"edgeAI_arch.h")

                f.write("\nenum {\n")
                f.write("HOR = " + str(int(N)) + ",\n")
                if self.opt.solver == "qp":
                    f.write("STATES = " + str(int(self.sys.nx)) + ",\n")
                    f.write("INPUTS = " + str(int(self.sys.nu)) + ",\n")
                    f.write("HOR_INPUTS = " + str(int(N*self.sys.nu)) + ",\n")
                    f.write("HOR_STATES = " + str(int(N*self.sys.nx)) + ",\n")
                    if self.sys.E.mat.size > 0:
                        f.write("DISTURBANCES = " + str(int(self.sys.nd)) + ",\n")
                        f.write("HOR_DISTURBANCES = " + str(int(N*self.sys.nd)) + ",\n")
                    if self.sys.F.mat.size > 0:
                        f.write("COUPLINGS = " + str(int(self.sys.nz)) + ",\n")
                        f.write("HOR_COUPLINGS = " + str(int(N*self.sys.nz)) + ",\n")
                    f.write("OPT_VAR = " + str(int((N+1)*self.sys.nx+N*self.sys.nu)) + ",\n")
                    f.write("MULT = " + str(int(self.prob.nm)) + ",\n")
                    f.write("MAXITER = " + str(int(self.opt.MAXITER)) + "\n};\n")

                elif self.opt.solver == "nn":
                    f.write("STATES = " + str(int(self.sys.nx)) + ",\n")
                    f.write("INPUTS = " + str(int(self.sys.nu)) + ",\n")
                    f.write("HOR_INPUTS = " + str(int(N*self.sys.nu)) + ",\n")
                    f.write("HOR_STATES = " + str(int(N*self.sys.nx)) + ",\n")
                    if self.sys.E.mat.size > 0:
                        f.write("DISTURBANCES = " + str(int(self.sys.nd)) + ",\n")
                        f.write("HOR_DISTURBANCES = " + str(int(N*self.sys.nd)) + ",\n")
                    if self.sys.F.mat.size > 0:
                        f.write("COUPLINGS = " + str(int(self.sys.nz)) + ",\n")
                        f.write("HOR_COUPLINGS = " + str(int(N*self.sys.nz)) + ",\n")
                    f.write("OPT_VAR = " + str(int((N+1)*self.sys.nx+N*self.sys.nu)) + "\n};\n")
                    # f.write("MULT = " + str(int(self.prob.nm)) + "\n};\n")

                self.wdtc(f)

        ### write c-file
        with open(path_c,"w") as f:

            if mode == "nn":

                self.witc(f,"edgeAI_const.h")

                self.nn.write_data(f)

                self.nn.write_struct(f)

                f.write("\nreal_t in[INPUT];\n")
                f.write("real_t in_scaled[INPUT];\n")
                f.write("real_t out[OUTPUT];\n")
                f.write("real_t out_scaled[OUTPUT];\n")

                f.write("\nstruct edgeAI_ctl ctl = {&dnn, in, in_scaled, out, out_scaled};\n")

            else:

                self.witc(f,"edgeAI_const.h")

                ### external data
                if self.ed:
                    self.ed.dist_vec.write(f)

                ## system
                if self.opt.gen_sim:
                    self.sys.write_data(f)

                ## optimization problem
                if self.opt.solver == "qp":
                    self.prob.write_data(f)
                ## neural networks
                elif self.opt.solver == "nn":
                    self.nn.write_data(f)

                # compute step sizes
                t = np.zeros([int(self.opt.MAXITER)+1,1])
                t[0] = 1
                for i in range(t.shape[0]-1):
                    t[i+1] = .5*(1+np.sqrt(1+4*t[i]**2))
                tdif = np.zeros([int(self.opt.MAXITER),1])
                for i in range(tdif.shape[0]):
                    tdif[i] = (t[i]-1)/t[i+1]
                tk = edgeAI_Matrix(tdif,"tk")
                tk.write(f)

                ### computation variables
                if self.sys.E.mat.size > 0:
                    f.write("real_t in[HOR_DISTURBANCES+DISTURBANCES];\n")
                    f.write("real_t d[DISTURBANCES];\n")

                f.write("\nreal_t x_trj[STATES+HOR_STATES];\n")
                f.write("real_t u_trj[HOR_INPUTS];\n")
                f.write("real_t x[STATES];\n")
                f.write("real_t u[INPUTS];\n")
                f.write("real_t z[OPT_VAR];\n")
                f.write("real_t mue[MULT];\n")

                ### predefintion of the structs
                if self.ed:
                    self.ed.write_struct(f)

                ctl_str = "\nstruct edgeAI_ctl ctl = {&data, &sys, "

                if self.opt.gen_sim:
                    self.sys.write_struct(f)

                if self.opt.solver == "qp":
                    self.prob.write_struct(f)
                    ctl_str += "&prob, "

                elif self.opt.solver == "nn":
                    self.nn.write_struct(f)
                    ctl_str += "&dnn, "

                f.write(ctl_str+"x_trj, u_trj, z, mue, x, u};\n")

    def write_makefiles(self):

        project_name = self.opt.project_name[0:-2]

        with open(self.opt.path+"/Makefile","w") as f:
            f.write("CC = gcc\n")
            f.write("FLAGS = -Os -Wall -Wstrict-prototypes -pedantic\n")
            f.write("OPT = -O3 -funroll-loops\n")
            f.write("STD = -std=c89\n")
            f.write("DIRS = edgeAI\n")
            f.write("GARBAGE_PATTERNS = *.o *.a\n")
            f.write("GARBAGE = $(foreach DIR,$(DIRS),$(addprefix $(DIRS)/,$(GARBAGE_PATTERNS)))\n\n")

            f.write("all: ledgeAI "+project_name+"\n\n")
            f.write("ledgeAI:\n\tmake -C edgeAI\n\n")
            f.write(project_name+": "+project_name+".c ledgeAI\n\t")
            f.write("$(CC) $(FLAGS) $(OPT) $(STD) "+project_name+".c -LedgeAI -ledgeAI -o "+project_name+" -lm\n\n")
            f.write("clean:\n\trm -rf "+project_name+" $(GARBAGE)")

        with open(self.opt.path+"/edgeAI/Makefile","w") as f:
            f.write("CC = gcc\n")
            f.write("FLAGS = -Os -Wall -Wstrict-prototypes -pedantic\n")
            f.write("OPT = -O3 -funroll-loops\n")
            f.write("STD = -std=c89\n\n")

            f.write("all: libedgeAI edgeAI_const.o edgeAI_main.o mtx_ops.o\n\n")
            f.write("libedgeAI: mtx_ops.o edgeAI_const.o edgeAI_main.o\n\t")
            f.write("ar rcs libedgeAI.a mtx_ops.o edgeAI_const.o edgeAI_main.o\n\n")
            f.write("edgeAI_const.o: edgeAI_const.c\n\t")
            f.write("$(CC) $(FLAGS) $(OPT) $(STD) -I./include -c edgeAI_const.c\n\n")
            f.write("edgeAI_main.o: edgeAI_main.c\n\t")
            f.write("$(CC) $(FLAGS) $(OPT) $(STD) -I./include -c edgeAI_main.c\n\n")
            f.write("mtx_ops.o: mtx_ops.c\n\t")
            f.write("$(CC) $(FLAGS) $(OPT) $(STD) -I./include -c mtx_ops.c\n\n")
            f.write("clean:\n\t")
            f.write("rm *.o libedgeAI.a\n")

    def wdtc(self,f,defname = None):
        """
            Write define to c-filename
        """
        if defname == None:

            f.write("\n#endif\n")

        else:

            f.write("#ifndef "+defname+"\n")
            f.write("#define "+defname+"\n")

    def witc(self,f,h_file):
        """
            Write bla
        """
        if not (h_file[0] == "<"):

            f.write("#include \"" + h_file + "\"\n")

        else:

            f.write("#include " + h_file + "\n")

    def wstc(self,f,name,kees,vartype,const):
        """
            A function writing structures
        """

        f.write("struct " + name + "{\n")
        for counter, kee in enumerate(kees):
            if const == 1:
                const_str = "const "
            else:
                const_str = ""

            if vartype[counter] == 1:
                var_str = "real_t "
            elif vartype[counter] == 2:
                var_str = "uint32_t "
            elif vartype[counter] == 3:
                var_str = "struct "

            f.write(const_str + var_str + kees[counter] + ";\n")

        f.write("};\n\n")

    def wftc(self,f,name,ivars,var_type,vts,var_size,se=None):
        """
            A function writing functions to c-Code
            Inputs:     f        - file handle
                        name     - name of the function (str)
                        ivars    - name of the input variables (list of strings)
                        var_type - type of the variable (e.g. real_t, uint32_t) (list of strings)
                        vts      - 1 -> const, 0 -> [] (list of binaries)
                        var_size - 1 -> size variable, 0 -> not variable (list of binaries)
                        se       - Type of function (static or extern) (str)
        """

        if se == "s":
            ftype = "static "
        elif se == "e":
            ftype = "extern "
        else:
            ftype = ""

        f.write("\n" + ftype + "void " + name + "(\n\t")
        for cnt, ivar in enumerate(ivars):

            if vts[cnt] == 1:
                vts_type = "const "
            else:
                vts_type = ""

            if var_size[cnt] == 1:
                vs_type = "[]"
            else:
                vs_type = ""

            f.write(vts_type + var_type[cnt] + " " + ivar + vs_type)

            if cnt == len(ivars)-1:
                f.write("\n\t);\n")
            else:
                f.write(",\n\t")

    def wftc_he(self,f,name=None,ivars=None,var_type=None,vts=None,var_size=None,se=None):

        if name == None:

            f.write("\n\n\treturn;\n}\n")

        else:

            if se == "s":
                ftype = "static "
            # elif se == "e":
            #     ftype = "extern "
            else:
                ftype = ""

            ### input variables
            f.write("\n" + ftype + "void " + name + "(")

            for cnt, var in enumerate(ivars):

                if vts[cnt] == 1:
                    vts_type = "const "
                else:
                    vts_type = ""

                f.write(vts_type + var_type[cnt] + " " + var)

                if cnt == len(ivars)-1:

                    f.write(")\n{\n\t")

                else:

                    f.write(", ")

    def wftc_lv(self,f,ivars,var_type):

        ### local variables
        var_type_str = []
        var_type_num = []
        for vt in var_type:

            if not vt in var_type_str:
                var_type_str.append(vt)

            var_type_num.append(var_type_str.index(vt))

        for cnt_str, vts in enumerate(var_type_str):

            f.write(vts + " ")
            idx = []
            for el in var_type:
                if el == vts:
                    idx.append(True)
                else:
                    idx.append(False)

            cnt_val = 0
            for cnt_idx, val in enumerate(idx):

                if val:
                    cnt_val += 1
                    if ((cnt_str == len(var_type_str)-1) and (cnt_val == sum(idx))):

                        f.write(ivars[cnt_idx] + ";\n\n\t")

                    elif (cnt_val == sum(idx)):

                        f.write(ivars[cnt_idx] + ";\n\t")

                    else:

                        f.write(ivars[cnt_idx] + ", ")

    def wfltc(self,f,start,stop,inc=None,var=None):
        """
            Write a for-loop to c
        """
        if type(start) == int:
            start = str(start)
        if type(stop) == int:
            stop = str(stop)
        if type(inc) == int:
            inc = str(inc)
        if inc == None:
            inc = "++"
        if var == None:
            var = "i"

        f.write("for (" + var + " = " + start + "; " + var + " < " + stop + "; " + var + inc + ") {\n")

    def mult(self,mat,mat_loc,vec,out,f):
        """
        mat: edgeAI_Matrix
        mat_loc: location of the matrix within struct(s)
        vec: multiplication vector name
        out: output vector name
        """
        if mat.sparse == True:
            f.write("mtx_times_vec_sparse(%s,%s_data,%s_ptr,%s_ind,%s,%s);\n" %\
                   (out,mat_loc+mat.name,mat_loc+mat.name,mat_loc+mat.name,vec,mat.vsr))
        else:
            f.write("mtx_times_vec_dense(%s,%s,%s,%s,%s);\n" %\
                   (out,mat_loc+mat.name,vec,mat.vsr,mat.vsc))

    def fpga_layer(self,layer,typ,f):
        self.mult_fpga(layer,typ,f)
        self.add_fpga(layer,typ,f)
        self.activation_fpga(layer,typ,f)

    def fpga_scale(self,typ,f):
        size = self.nn.kernel[0].cols
        if typ == "h":
            f.write("extern void mult_scale(real_t pout[%i], const real_t pin[%i], const real_t psca[%i]);\n\n" % (size,size,size))
            f.write("extern void mtx_substract_scale(real_t pmtxc[%i], const real_t pmtxa[%i], const real_t pmtxb[%i]);\n\n" % (size,size,size))
        elif typ == "c":
            f.write("\nvoid mult_scale(real_t pout[%i], const real_t pin[%i], const real_t psca[%i])\n" % (size,size,size))
            f.write("{\n\t")
            self.wftc_lv(f,["k"],["uint32_t"])
            self.wfltc(f,0,str(size),var="k")
            f.write("\t\tpout[k] = pin[k] * psca[k];\n")
            f.write("\t}\n")
            self.wftc_he(f)
            f.write("\nvoid mtx_substract_scale(real_t pmtxc[%i], const real_t pmtxa[%i], const real_t pmtxb[%i])\n" % (size,size,size))
            f.write("{\n\t")
            self.wftc_lv(f,["k"],["uint32_t"])
            self.wfltc(f,0,str(size),var="k")
            f.write("\t\tpmtxc[k] = pmtxa[k] - pmtxb[k];\n")
            f.write("\t}\n")
            self.wftc_he(f)

    def fpga_unscale(self,typ,f):
        size = self.nn.kernel[-1].rows
        if typ == "h":
            f.write("extern void mult_unscale(real_t pout[%i], const real_t pin[%i], const real_t psca[%i]);\n\n" % (size,size,size))
            f.write("extern void mtx_add_unscale(real_t pmtxc[%i], const real_t pmtxa[%i], const real_t pmtxb[%i]);\n\n" % (size,size,size))
        elif typ == "c":
            f.write("\nvoid mult_unscale(real_t pout[%i], const real_t pin[%i], const real_t psca[%i])\n" % (size,size,size))
            f.write("{\n\t")
            self.wftc_lv(f,["k"],["uint32_t"])
            self.wfltc(f,0,str(size),var="k")
            f.write("\t\tpout[k] = pin[k] * psca[k];\n")
            f.write("\t}\n")
            self.wftc_he(f)
            f.write("\nvoid mtx_add_unscale(real_t pmtxc[%i], const real_t pmtxa[%i], const real_t pmtxb[%i])\n" % (size,size,size))
            f.write("{\n\t")
            self.wftc_lv(f,["k"],["uint32_t"])
            self.wfltc(f,0,str(size),var="k")
            f.write("\t\tpmtxc[k] = pmtxa[k] + pmtxb[k];\n")
            f.write("\t}\n")
            self.wftc_he(f)

    def mult_fpga(self,layer,typ,f):
        mat = self.nn.kernel[layer]
        if typ == "h":
            f.write("extern void mtx_times_vec_dense_%i(real_t pout[%i], const real_t pmtx[%i], const real_t pvec[%i]);\n\n" % (layer+1,mat.rows,mat.rows*mat.cols,mat.cols))
        elif typ == "c":
            f.write("\nvoid mtx_times_vec_dense_%i(real_t pout[%i], const real_t pmtx[%i], const real_t pvec[%i])\n" % (layer+1,mat.rows,mat.rows*mat.cols,mat.cols))
            f.write("{\n\t")
            self.wftc_lv(f,["i","j","k = 0"],["uint32_t","uint32_t","uint32_t"])
            self.wfltc(f,0,str(mat.rows))
            f.write("\t\tpout[i] = 0;\n")
            f.write("\t\t")
            self.wfltc(f,0,str(mat.cols),var="j")
            f.write("\t\t\tpout[i] += pmtx[k] * pvec[j];\n")
            f.write("\t\t\tk++;\n")
            f.write("\t\t}\n")
            f.write("\t}\n")
            self.wftc_he(f)

    def add_fpga(self,layer,typ,f):
        mat = self.nn.bias[layer]
        size = mat.rows
        if typ == "h":
            f.write("extern void mtx_add_%i(real_t pmtxc[%i], const real_t pmtxa[%i], const real_t pmtxb[%i]);\n\n" % (layer+1,size,size,size))
        elif typ == "c":
            f.write("\nvoid mtx_add_%i(real_t pmtxc[%i], const real_t pmtxa[%i], const real_t pmtxb[%i])\n" % (layer+1,size,size,size))
            f.write("{\n\t")
            self.wftc_lv(f,["k"],["uint32_t"])
            self.wfltc(f,0,str(size),var="k")
            f.write("\t\tpmtxc[k] = pmtxa[k] + pmtxb[k];\n")
            f.write("\t}\n")
            self.wftc_he(f)

    def activation_fpga(self,layer,typ,f):
        mat = self.nn.bias[layer]
        if typ == "h":
            if self.nn.act_fun[layer] == "relu":
                f.write("extern void mtx_relu_%i(real_t vec[%i]);\n\n" % (layer+1,mat.rows))
            elif self.nn.act_fun[layer] == "tanh":
                f.write("extern void mtx_tanh_%i(real_t vec[%i]);\n\n" % (layer+1,mat.rows))
        elif typ == "c":
            size = self.nn.bias[layer].rows
            if self.nn.act_fun[layer] == "relu":
                f.write("\nvoid mtx_relu_%i(real_t vec[%i])\n" % (layer+1,mat.rows))
                f.write("{\n\t")
                self.wftc_lv(f,["k","zero = 0."],["uint32_t","real_t"])
                self.wfltc(f,0,str(size),var="k")
                f.write("\t\tif (vec[k] < zero) {\n")
                f.write("\t\t\tvec[k] = zero;\n\t\t}\n")
                f.write("\t}\n")
                self.wftc_he(f)
            elif self.nn.act_fun[layer] == "tanh":
                f.write("\nvoid mtx_tanh_%i(real_t vec[%i])\n" % (layer+1,mat.rows))
                f.write("{\n\t")
                self.wftc_lv(f,["k"],["uint32_t"])
                self.wfltc(f,0,str(size),var="k")
                f.write("\t\tvec[k] = tanh(vec[k]);\n")
                f.write("\t}\n")
                self.wftc_he(f)
