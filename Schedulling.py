'''
Created on 13 Sep 2020
Schedulling v0
We will suppose here that we can have several items with the same mold ! But not otherwise !!
@author: Uri-J
'''

import logging
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import mip
from mip import Model, xsum, minimize, maximize, OptimizationStatus, BINARY, INTEGER
import time

import matplotlib.pyplot as plt

"""


 
"""


"""
@see: 

I: Different items
J: Molds
M: Machines


d: vector (1, n_items) Demands of each item (size 1, I) (int)
omega: vector (1, n_items) How much more it cost to buy a piece on another site. Can be set to a very high number if production might be exact (float32)
M: binary Matrix (n_machine, n_molds) setting  machine-mold relation.  (bool)
I: Binary Matrix (n_items, n_molds) setting items-mold relation.(bool)
st: Matrix of times: (n_items, n_molds) // piece i setup time with mold j // )
it: Matrix of times : (n_machine, n_molds) instalation time of mold j on machine k (float32)
dt: Matrix of times : (n_machine, n_molds) representing the removal time of mold j on machine k (float32)
vij: Matrix of inverse of time (n_items, n_molds) (float32)
tm: Vector (1, n_machines); Avaible time for production. i.e.  depending on mantenance plans  (float32)
v_ijk: 3D-array of time/piece  (n_items, n_molds, n_machine): Production time of piece i, using mold j and machine k !!


# Xijk amount of pieces i, to be produced with mold j on machine k. The objective function to maximize is:

sum_items omega_i sum_molds sum _machines X_{ijk}

Variables number: n_items*n_machines*n_molds


""" 

zj = {} # Values of z with mold j
jk = {} # Values of j with machine k
iz = {} # Items i ->  z
ik = {} # Items i used in machine k
zk = {} # Values of z with machine k
kz = {} # Machines k that can do a product z
kj = {} # Machines k that can use mold j
ij = {} # Set of pieces i that can be produced using mold j
ji = {} # Set of molds j that can be used for producing piece i ## Normally 1 per mold
reduction = pd.DataFrame()
global Time_mean
global Time_variance 

def size_assert(array, lenght):
    if (isinstance(lenght, int)):
        lenght = np.array([lenght])
        array = array.reshape(int(lenght),1)
        
    if(array.ndim == len(lenght)):
        for i in range(0, array.ndim):          
            assert(array.shape[i] == lenght[i])    
    return True

def hypotheses(I):
    
    # Only one item per mold.
    for i in range(0,I.shape[0]):
        if(sum(I[i][:])>1):
            return False
    
    
    return True


def reduction_Dataframe(I, columns):
    # Number of different pieces types a mold can do
    reduction = pd.DataFrame(columns = columns)
    z = 0
    for i in range(0,I.shape[0]):
        for j in range(0, I.shape[1]):
            if(I[i][j] == 1):
                reduction = reduction.append({"z": z, columns[0]: i, columns[1]: j}, ignore_index = True)
                z+=1
    
    return reduction
    
    
def expand(X, reduction):
    # i : item
    # j: Mold
    # k: Machine
    # X: matrix(Z, K) -> (I,J,K)
    if(X.ndim == 1 or (X.shape[0] == 1 or X.shape[1] == 1)):
        # Typically omega(i) -> omega(z)
        y = np.zeros((len(reduction.index)))
        for z in reduction.index:
            i = reduction.loc[z,"Item"]
            y[z] = X[i]
        
        return y
    elif(X.ndim == 2):
        mold_max = np.max(reduction["Mold"])
        item_max = np.max(reduction["Item"])
        Y = np.zeros((item_max, mold_max, X.shape[1]))
        
        for k in range(0, X.shape[1]):
            for z in reduction.index:
                i, j = reduction.loc[z,["Item", "Mold"]]
                Y[i][j][k] = X[z][k]
        return Y
        
    else:
        logging.warning("You are trying to expand a matrix that has dimension 3 !!!")
        return None

def shrink(X,reduction):
    #X: matrix(I,J, K) -> (Z,K)
 # i : item
    # j: Mold
    # k: Machine
    # X: matrix(Z, K) -> (I,J,K)
    if(X.ndim == 1):
        logging.warning("You are trying to shrink a 1-Dimensional array !! It's not possible")
        return None
    elif(X.ndim == 2):
        logging.warning("2D shrink Not implemented, yet")        
        return None
    elif(X.ndim == 3):
        
        Z = len(reduction.index)
        Y = np.zeros((Z, X.shape[2]))
        
        for k in range(0, X.shape[2]):
            for z in reduction.index:
                i, j = reduction.loc[z,["Item", "Mold"]]
                Y[z][k] = X[i][j][k]  
        return Y                       
    else:
        logging.warning("How you got a 4+-Dimensional array ?????? !!!")
        return None



def sparse_binary_matrix(X, onesize):
    if(onesize>= 1):
        logging.warning("One-size value has to be less than one !")
        return 0
    
    nx,ny = X.shape[0], X.shape[1]
    X = np.zeros((nx,ny))
    for i in range(0, nx):
        dx = int(np.round((ny-1)*np.random.rand()))
        X[i][dx] =1
        while(np.random.rand()<onesize):
            dx2 = int(np.round((ny-1)*np.random.rand()))             
            X[i][dx2] =1
   
    return X

def Z(j, reduction):
    z = reduction[reduction["Mold"]==j].index.values.tolist()
    return z
    
    
    #i, j = reduction.loc[z,["Item", "Mold"]]
    

def variables_relations(I, M):   
        
    global zj  # Values of z with mold j
    global jk  # Values of j with machine k
    global iz  # Items i ->  z
    global ik  # Items i used in machine k
    global zk  # Values of z with machine k
    global kz  # Machines k that can do a product z
    global kj  # Machines k that can use mold j
    global ij  # Set of pieces i that can be produced using mold j
    global ji  # Set of molds j that can be used for producing piece i ## Normally 1 per mold
    
    for j in J:
        zj[j] = reduction[reduction["Mold"]==j].index.values.tolist()
    
    for k in K:
        jk[k] = moldMachine[moldMachine["Machine"] == k]["Mold"].values.tolist()
    
    for j in J:
        kj[j] = moldMachine[moldMachine["Mold"] == j]["Machine"].values.tolist()
        
    for z in Z:
        iz[z] = reduction.loc[z, "Item"]
        
    for k in K:
        mold_values = jk[k]
        list_values = []
        for j in mold_values:
            list_values.extend(zj[j])
        zk[k] = list(dict.fromkeys(list_values))
        
    list_z = {}
    
    for z in Z:
        list_z[z] = []
    for z in Z:
        for k in K:
            if (z in zk[k]):
                list_z[z].append(k)
    
    for z in Z:
        kz[z] = list(dict.fromkeys(list_z[z]))
    
    for k in K:
        val = zk[k]
        list_k = []
        for z in val:
            list_k.append(iz[z])
        ik[k] = list(dict.fromkeys(list_k))

    
    for j in J:
        ij[j] = []
        for i,x in enumerate(I[:,j]):
            if(x == 1):
                ij[j].append(i*x)
        
    for i in reduction["Item"].values.tolist():
        ji[i] = []
        for j, x in enumerate(I[i, :]):
            if (x == 1):
                ji[i].append(j*x)       
                
    return True

def init(parameters):
    
    machines = parameters['machines']
    molds = parameters['molds'] 
    items = parameters["items"] 
    d = parameters["d"] 
    omega = parameters["omega"] 
    M = parameters["M"]
    I = parameters["I"]
    st = parameters["st"] 
    V = parameters["V"] 
    it = parameters["it"] 
    dt = parameters["dt"] 
    tm = parameters["tm"]
    dtype = parameters["dtype"] 
    
    """ Set of variables """
    global J,K,Items
    J = set(range(0, molds))
    K = set(range(0, machines))
    Items = set(range(0, items))
    
    
    """ Size and type verification """
    
    
    d = d.astype(np.int32)
    M = M.astype(np.bool)
    I = I.astype(np.bool)
    omega = omega.astype(dtype)
    st = st.astype(dtype)
    it = it.astype(dtype)
    dt = dt.astype(dtype)
    tm = tm.astype(dtype)
    V = V.astype(dtype) 

    size_assert(d, items)
    size_assert(omega, items)
    size_assert(M, [molds, machines])
    size_assert(I, [items, molds])
    size_assert(st, [items, molds])
    size_assert(it, [molds, machines])
    size_assert(dt, [molds, machines])
    size_assert(tm, machines)   
    size_assert(V, [items, molds, machines])
    
    print(" All array dimensions seems correct :)")
    
   
    

    
    #n_variables = items*molds*machines ## Is too much, and most are zero !!!! We need a reduction of variables. 
    # we shrink variable (i,j) into z. As we know the relational matrix I. 
    # We will suppose that we have at least one mold per item
    
    """ Hypotheses: """
    
    if(not hypotheses(I)):
        logging.warning("The hypotheses for this study are not valid !!  Here we can do the same item with diferents molds")
        return 0
    

    """ Data reduction """
    global reduction, moldMachine
    reduction = reduction_Dataframe(I, ["Item", "Mold"])
    moldMachine = reduction_Dataframe(M, ["Mold", "Machine"])
    
    global Z
    Z = reduction.index
    
    
    variables_relations(I, M)
    reduction = reduction_Dataframe(I, ["Item", "Mold"])
    reduction.to_csv("reduction.csv")
    
    
    #print(moldMachine)
    # We suppose len(Z) == len(d) !! It's the hypothesis ! If not, it is doable but harder :(
    
    if(len(Z)!=len(d)):
        logging.warning("len(Z) != len(d) is breaking the hypotheses we have set :(  ")
        return 0
    
    
    # omega(items) -> omega (z)  demand(items) -> demand(z)
    
    omega = expand(omega, reduction)
    d = expand(d, reduction)
    
    # V(i,j,k) -> V(z,k)
    
    V = shrink(V, reduction)
    
    d, omega, st, it, dt, tm, V,
    
    parameters = {"d":d, "omega": omega, "st":st, "it":it, "dt": dt, "tm": tm, "V": V, "M": I, "I": I}
    
    
    
    return parameters
    
""" Linear Programmation problem: """

def LPModel(parameters_model,  hard_mold_constraint, dtype):
    
    
    
   
    d = parameters_model["d"]
    omega = parameters_model["omega"]
    st = parameters_model["st"]
    it = parameters_model["it"]
    dt = parameters_model["dt"]
    tm = parameters_model["tm"]
    V = parameters_model["V"]
    
    
    
    machines = len(K)
    molds = len(J)
    items = len(Z)
    average_time_production = np.average(V) # 3seg/piece
    t_optimal = average_time_production*np.sum(d)/(machines+1)

    """ Model """
    
    mold_time = it+dt  # The time to remove and insert a new mold:
    tm_max = np.max(tm) # Maximum avaible time between all machines
    
    
    model = Model()
    # y: Production value of piece-mold z and machine k
    y = [[model.add_var(name = "y({},{})".format(z,k), var_type = INTEGER) for k in K] for z in Z]
    
    #binary variables indicating if mold replacement(j,k) is used or not: b, n
    b = [[model.add_var(name = "b({},{})".format(z,k) ,var_type =BINARY) for k in K] for z in Z]
    n = [[model.add_var(name = "n({},{})".format(j,k), var_type = BINARY) for k in K] for j in J]

    
    """ Function to maximize"""
    
    model.objective = maximize(xsum(omega[z]*y[z][k] for z in Z for k in K))
    #model.objective = minimize(xsum((xsum(n[j][k]*mold_time[j][k] for j in J for k in K)+xsum(V[z][k]*y[z][k]+b[z][k]*st[z][k] for z in Z f))))
    
    
    """ Restrictions """
    
    """ ----------------- c0 eliminating impossibles values ---------- """
    
    for k in K:
        for j in J:
            if (j not in jk[k] or k not in kj[j]):
                model += n[j][k] == 0
        for z in Z:
            if(z not in zk[k] or k not in kz[z]):
                model += b[z][k] == 0
                model += y[z][k] == 0
            else:
                model += y[z][k] >= 0
                   
    
    """ ----------------- c1: Production on each z (machine-mold) <= demand*[1 or 0] depending if we use the machine or not ----------    I*J equations  """
    
    for z in Z:
        for k in kz[z]:
            model += y[z][k] - d[z]*b[z][k] <=  0  

     
    
    """ ----------------- c3: Production on each all machines <= demand_item ----------               I equations  """
    for z in Z:
        model += xsum(y[z][k] for k in kz[z]) <= d[z]    
    
    """ ----------------- c4: Time of installation/removal of a mold (left) + time producing + set up time(using the same machine/mold)  
                              is lower than the available time of the machine: actual_time - manteinance_time    --------------               k equations  """
    
    for k in K:    
        model += xsum(n[j][k]*mold_time[j][k] for j in jk[k])+xsum(V[z][k]*y[z][k]+b[z][k]*st[z][k] for z in zk[k]) <= tm[k] 
        
    """ ----------------- c5: Time of installation/removal of a mold (left) + time producing + set up time(using the same machine/mold)  
                              is lower than the maximum available time of the machine: actual_time - manteinance_time    --------------               J equations  """
        
    for j in J:
        model += xsum(n[j][k]*mold_time[j][k] for k in kj[j]) +xsum(y[z][k]*V[z][k] +b[z][k]*st[z][k] for z in zj[j]) <= tm_max
        

 
    #for k in K:
     #   model += xsum(y[z][k]*V[z][k] for z in zk[k])- xsum(n[j][k]*mold_time[j][k] for j in jk[k]) >= 0
    """ ----------------- c5 (opt): Each machine uses one mold ----------   We are not sure that is a good equation            J equations  """ 
    if (hard_mold_constraint):
        for j in J:
            model += xsum(n[j][k] for k in kj[j]) <=1
      
    
    
    """ ---------------- c6: Assign mold j to machine k if there is at least one piece produced with this mold-machine pair, regardless of the type of piece--------- I*J equation """
    
    
    for k in K:
        for j in jk[k]:
            l = len(ij[j]) # Number of pieces z that can be obtained with mold j
            model += xsum(b[z][k] for z in zj[j]) <= l*n[j][k] 
            
            
    
    """ ----------------- c7 (opt): ----------               I*J  equations  """
    """for k in K:
        for j in jk[k]:
            model += n[j][k] <= M[j][k]"""
    
    """ ---------------- c8 (opt) -------------------- I*J equation """
    
    """for k in K:
        for z in zk[k]:
            j = reduction.loc[z,"Mold"]  
            model += b[z][k] <= M[j][k]"""
            
    
    """ ---------------- c9 (opt) -------------------- I*J equation """
    
    """for k in K:
        for j in jj[k]:           
            if(len(zz[j])>0):
                    model += xsum(y[z][k] for z in zz[j]) <= xsum(n[j][k]*d[z] for z in zz[j])"""
                    

    """ ---------------- c10 (opt) -------------------- k equation """        

            
    """ for k in K:    
        model += xsum(n[j][k]*mold_time[j][k] for j in jj[k])+xsum(V[z][k]*y[z][k]+b[z][k]*st[z][k] for z in zk[k]) >= 1 """
    
     # Time variable:
    Tj = np.zeros((items, machines), dtype = dtype)
    
    
    model.preprocess = 0
    model.max_solutions = 100
   
    model.seed = parameters_model['seed']
    status = model.optimize(max_seconds_same_incumbent = 50 )
     
    if status == OptimizationStatus.OPTIMAL:
        
        print("Solution with cost: {}".format(model.objective_value))
        print("Solutions founded: {}".format(model.num_solutions))
    
       
    
        for k in K:
            for z in zk[k]:
                j = ji[z][0] # Only one mold per z
                Tj[z,k] = n[j][k].x*mold_time[j,k]+y[z][k].x*V[z][k]+b[z][k].x*st[z][k]
                
                #Tj[z,k] = y[z][k].x*V[z][k]
    else: 
        logging.warning("Simulation failed !! I can't find any solution for this problem :( ")
  

    X = np.zeros((items, machines))
    B = np.zeros((items, machines))
    N = np.zeros((molds, machines))
    
    for k in K:
        for z in Z:
            B[z,k] = b[z][k].x
            X[z,k] = y[z][k].x
        for j in J:
            N[j,k] = n[j][k].x
    
    results = {"B": B, "X": X, "N": N, "Tj": Tj}
    
    return results


def show_results(parameters, results):
    
    b = results["B"]
    y = results["X"]
    n = results['N']
    Tj = results["Tj"]
    
    d = parameters["d"]
    
    production = pd.DataFrame()   
    Tk = np.sum(Tj, axis = 0)
                         
    """ Results """


    sumb = 0
    print("Schedulle ") 
    for k in K:
        print("Machine "+str(k)+ " with time " + str(Tk[k]/3600)+"h")
        print("Can use the following molds: "+ str(jk[k]))
        for z in Z:
            i, j = reduction.loc[z,["Item", "Mold"]]
            if y[z,k]>0:
                sumb += 1
                production = production.append({"Machine": k, "Item": i, "Mold": j, "#pieces": y[z, k],  "Time": Tj[z,k]}, ignore_index = True)
                print("Item: "+ str(i)+ ", Mold: "+str(j)+ ", # pieces: "+ str(y[z, k]) + " and time: "+str(Tj[z,k]/3600)+"h")
    
    total_demand = np.sum(d)
    total_production = np.sum(np.sum(y))
    print("\nTotal demand: "+str(total_demand))
    print("Total Production: "+str(total_production))
    
    prod_machine = np.zeros((len(K)))
    prod_items = np.zeros((len(Z)))
    
    
    
    prod_items = np.sum(y, axis = 1)
    
    prod_machine = np.sum(y, axis = 0)
    
    if(total_production<total_demand):
        prod_diference = d - prod_items
        for z,x in enumerate(prod_diference):
            if(x>0):
                print("We can't produce "+str(x)+ " unities of item "+str(iz[z])+ " with the following molds: "+str(ji[iz[z]]))
        
    
    
    print("sum of n: "+str(np.sum(np.sum(n, axis = 0))))
    print("sum of b: "+str(np.sum(np.sum(b, axis = 0)))+" vs "+ str(sumb))
    
    production.to_csv("production.csv")

    
    return 0



# Mold Overlapping Detection
# Finds if there is any mold overlapping, and if we can find some time intervals for not overlapping. 
# If it is not possible

def MOD(results):
    n = results['N']
    mod = False
    overlapping = []
    for j in J:
        mold_sum = 0
        for k in kj[j]:
            mold_sum = n[j][k]
            
        if mold_sum>1:
            mod = True 
            overlapping.append(j)
    
    if (not mod):
        return False
    else: 
    
        b = results["B"]
        y = results["X"]
   
        Tj = results["Tj"]
    
    return None





def Analyse_results(Best, current):
    
    
    Tj = current["Tj"]
    Tj_best = Best['Tj']
    
    
    if (np.mean(Tj_best) == 0):
        return current

    else:
        Tk = np.sum(Tj, axis = 0)
        Tk_best = np.sum(Tj, axis = 0)
    
        current_var = np.var(Tk)
        best_var = np.var(Tk_best)
    
        if(current_var<best_var):
            return current
        else:
            return Best





parameters = {}

## Example 

n_machine = 50
n_mold = 100
n_items = 110  # Items > mold always

parameters['machines'] = n_machine
parameters['molds'] = n_mold
parameters["items"] = n_items

parameters["d"] = np.round(300000*np.random.rand(n_items))
parameters["omega"] = 5*np.ones(n_items)

parameters["M"] = sparse_binary_matrix(np.zeros((n_mold, n_machine)),0.7 )
parameters["I"] = sparse_binary_matrix(np.zeros((n_items, n_mold)), 0)
 

parameters["st"] = 100*np.ones((n_items,n_mold)) # 5min set-up
parameters["V"] = 3*np.ones((n_items, n_mold, n_machine)) # 3 sec / piece
parameters["it"] = 3600*np.ones((n_mold, n_machine)) # 1h each operation
parameters["dt"] = 1800*np.ones((n_mold, n_machine)) # 30min each operation
parameters["tm"] = 3000000*np.ones((n_machine))

parameters["dtype"] = np.float32
## We make sure all the variables have the good size, the good format and that they follow the hypotheses of this problem.

parameters_model  = init(parameters)


## We use a decomposition approach. On the first part we use a Linear Programmation model:
feasible =True
parameters_model['seed'] = 0

B, X, N, Tj = [{} for _ in range(4)]
hard_mold_constraint = True
BestResults = {}
BestResults['Tj'] = 0
BestResults['B'] = B
BestResults['N'] = N
BestResults['X'] = X
num_it = 3
Time_mean = np.zeros((num_it))
Time_variance = np.zeros(())

while(feasible):
    
    
    for i in range(0,num_it):
        parameters_model['seed'] = i
        LP_results = LPModel(parameters_model,  hard_mold_constraint, np.float32)
        BestResults = Analyse_results(BestResults, LP_results)
        Time_machine = np.sum(LP_results['Tj'],axis = 0)
        Time_mean[i] = np.mean(Time_machine)
        Time_variance[i] = np.var(Time_machine)
    show_results(parameters_model, BestResults)
    #LP_results2 = LPModel(parameters_model,  False, np.float32)
    #show_results(parameters_model, LP_results2)

    # Wait here for the result
    """ #Feasibility problem Mold Overlapping Detection MOD """
    feasible  = False # MOD(LP_results)



plt.plot(range(0,num_it), Time_mean)

















"""

class work:
    machine = -1
    mold = -1
    item = -1
    production = 0
    time = 0
class machine:
    number = -1
    molds = []
    items = []
    prduction_speed = -1 # Might be a vector of mold too

class mold:
    number = -1
    machines = []
    items = []
    
    
"""

    