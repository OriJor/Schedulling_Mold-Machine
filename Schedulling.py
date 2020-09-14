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
from itertools import product

"""


 
"""

def size_assert(array, lenght):
    if (isinstance(lenght, int)):
        lenght = np.array([lenght])
        array = array.reshape(int(lenght),1)
        
    if(array.ndim == len(lenght)):
        for i in range(0, array.ndim):          
            assert(array.shape[i] == lenght[i])    
    return True
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
def hypotheses(I):
    
    # Only one item per mold.
    for i in range(0,I.shape[0]):
        if(sum(I[i][:])>1):
            return False
    
    
    return True


def reduction_Dataframe(I, columns):
    # Number of different pieces types a mold can do
    reduction = pd.DataFrame(columns = columns)
    
    for i in range(0,I.shape[0]):
        for j in range(0, I.shape[1]):
            if(I[i][j] == 1):
                reduction = reduction.append({columns[0]: i, columns[1]: j}, ignore_index = True)
    
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

def init(items, molds, machines, d, omega, M, I, st, it,dt, tm, V, dtype):
    
    J = set(range(0, molds))
    K = set(range(0, machines))
    Items = set(range(0, items))
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
    
    print(" All arrays dimensions seems correct :)")
   
        
   
    
    #n_variables = beta.shape[0]*beta.shape[1]
    
    
    #n_variables = items*molds*machines ## Is too much, and most are zero !!!! We need a reduction of variables. 
    # we shrink variable (i,j) into z. As we know the relational matrix I. 
    # We will suppose that we have at least one mold per item
    
    if(not hypotheses(I)):
        logging.warning("The hypotheses for this study are not valid !!  Here we can do the same item with diferents molds")
        return 0
    
    
    reduction = reduction_Dataframe(I, ["Item", "Mold"])
    
    moldMachine = reduction_Dataframe(M, ["Mold", "Machine"])
    
    
    
    # omega(items) -> omega (z) 
    omega = expand(omega, reduction)
    # V(i,j,k) -> V(z,k)
    V = shrink(V, reduction)
    model = Model()
    Z = reduction.index
    
    # y: Production value of piece-mold z and machine k
    y = [[model.add_var(name = "y({},{})".format(z,k), var_type = INTEGER) for k in K] for z in Z]
    
    #binary variables indicating if mold replacement(j,k) is used or not: b, n
    b = [[model.add_var(name = "b({},{})".format(z,k) ,var_type =BINARY) for k in K]for z in Z]
    n = [[model.add_var(name = "n({},{})".format(j,k), var_type = BINARY) for k in K] for j in J]
    
    model.objective = maximize(xsum(omega[z]*y[z][k] for z in Z for k in K))
    
    
    ## We suppose len(Z) == len(d) !! It's the hypothesis ! If not, it is doable but harder :(
    
    if(len(Z)!=len(d)):
        logging.warning("len(Z) != len(d) is breaking the hypotheses we have set :(  ")
        return 0
    
    
    for z in Z:
        for k in K:
            model += y[z][k] >= 0
    
    for z in Z:
        model += xsum(y[z][k] for k in K) <= d[z]   
    
    for z in Z:
        for k in K:
            model += y[z][k] <= d[z]*b[z][k]
    
    
    for k in K:    
        model += xsum(n[j][k]*(it[j][k]+dt[j][k]) for j in J)+xsum(V[z][k]*y[z][k]+ b[z][k]*st[z][k] for z in Z) <=tm[k]
    
    
    for z in Z:
        for k in K:
            model += b[z][k] <= xsum(I[z][j]*n[j][k] for j in J)             
            
    
    model.optimize()
    
    
    print("Total demand: "+str(np.sum(d)))
    
    
    
    sum = 0
    for k in K:
        for z in Z:
           sum+=y[z][k].x
           
    print("Total Production ", sum) 
    
    print("Schedulle ", sum) 
    for k in K:
        print("Machine "+str(k)+ ":")
        for z in Z:
            i, j = reduction.loc[z,["Item", "Mold"]]
            if n[j][k]>=0.99:
                print("Item: "+ str(i)+ ", Mold: "+str(j)+ ", # pieces: "+ str(y[z][k].x))
    """
    for k in K:
        for z in Z:
            print("Value b({}, {}): ".format(z,k)+str(b[z][k].x))
            
    for k in K:
        for j in J:
            print("Value n({}, {}): ".format(j,k)+str(n[j][k].x))
"""
    ## Bound

    
    


    


    
    return 0















## Example 

n_machine = 3
n_mold = 5
n_items = 6
#Item by mold

#d = np.array([1000, 20000, 5000, 300, 7000, 12000])
d = 30000*np.random.rand(n_items)
#omega = 10*np.ones(len(d))
omega = np.random.rand(n_items)+1.2
omega = 5*np.ones(n_items)
np.random.seed(3)
M = np.round(np.random.rand(n_mold, n_machine))

# Mold        0  1  2  3  4
# Machine 0: [1. 1. 0. 1. 1.]
# Machine 1: [1. 0. 0. 0. 0.]
# Machine 2: [0. 0. 1. 0. 1.]
 
# Machine 1 only can work with mold 0! 

I = np.array([[1,0,0,0,0],
             [0,1,0,0,0],
             [0,0,1,0,0],
             [0,0,0,1,0],
             [0,0,0,0,1],
             [0,0,0,0,1]])



# Mold     0  1  2  3  4
# Item 0: [1. 1. 0. 1. 1.]
# Item 1: [1. 0. 0. 0. 0.]
# Item 2: [0. 0. 1. 0. 1.]
# Item 3: [1. 0. 0. 0. 0.]
# Item 4: [0. 0. 1. 0. 1.]
# Item 5: [0. 0. 1. 0. 1.]

#Almost diagonal, mold 4 can produce items 4 and 5 (different colors diference)


st = 30*np.ones((n_items,n_mold)) # 5min set-up
V = 3*np.ones((n_items, n_mold, n_machine)) # 3 sec / piece
it = 3600*np.ones((n_mold, n_machine)) # 1h each operation
dt = 1800*np.ones((n_mold, n_machine)) # 30min each operation
tm = 360000*np.ones((n_machine))

init(n_items, n_mold, n_machine, d, omega, M, I, st, it,dt, tm, V, np.float32)

    