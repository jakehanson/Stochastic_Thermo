import itertools
import numpy as np
import pandas as pd

## Define a function to calculate the non-equilibrium free energy for a given distribution
def get_F(spin_configs,p_dist,M,J,beta):

	E_avg = 0.
	for i in range(len(spin_configs)):
		config = spin_configs[i]
		p_i = p_dist[i]
		E_avg = E_avg+p_i*get_H(config,M,J)

	S = get_S(p_dist)
	return(E_avg-S/beta)


## Define a function to calculate the hamiltonian for a given spin config
def get_H(spin_config,M,J):
    length = len(spin_config)
    
    H_tot = 0.
    for i in range(length-1):
        H_i = 0.
        if i%2 == 0:
            H_i = -M*spin_config[i]
        else:
            H_i = -J*(spin_config[i]*spin_config[i-1]+spin_config[i]*spin_config[i+1])-M*spin_config[i]
        H_tot = H_tot + H_i    
        
    ### Calculate contribution from PBC
    H_i = 0
    if length%2 == 0:
        H_i = -J*(spin_config[-1]*spin_config[-2]+spin_config[-1]*spin_config[0])-M*spin_config[-1]
    else:
        H_i = -M*spin_config[-1]
    H_tot = H_tot + H_i

    return(H_tot)


## Define a function to return all spin configs of size n
def all_configs(n):
    final_array = []
    for i in itertools.product('01', repeat=n):
        empty_list = []
        for each in i:
            if each == '0':
                empty_list.append(-1)
            else:
                empty_list.append(1)
        final_array.append(empty_list)
    return(final_array)
    
    for i in range(np.power(2,n)):
        print(i)

## Define function to calculate the actual EP
def get_EP(W,p_array,dt):
    EP = 0.
    n,m = np.shape(W)
    for i in range(n):
        for j in range(m):
            EP = EP + 1/2.*(W[i,j]*p_array[j]-W[j,i]*p_array[i])*np.log(W[i,j]*p_array[j]/(W[j,i]*p_array[i]))*dt
    return(EP)

## Define function to calculate the actual EF
def get_EF(W,p_array,dt):
    EF = 0.
    n,m = np.shape(W)
    for i in range(n):
        for j in range(m):
            EF = EF + 1/2.*(W[i,j]*p_array[j]-W[j,i]*p_array[i])*np.log(W[j,i]/W[i,j])*dt
    return(EF)


## Define Function to calculate entropy
def get_S(dist):
    S = 0
    for p_i in dist:
        if p_i != 0.:
            S = S - p_i*np.log(p_i)
    return(S)


## Define Function to get the units in our autonomous spin model
def get_units(n):
    if n%2 != 0:
        print("ERROR - n must be even!")
        return(-1)
    else: 
        units = []
        for i in range(n-1):
            if i%2 == 0:
                units.append([i])
            else:
                units.append([i-1,i,i+1])
        units.append([n-2,n-1,0])
        return(units)

    
## Define Function to calculate in-ex sum for a given dependency struct
def I_Nstar(p_array,units,n):
    prob_dist = p_array
#     print("\nPROB DIST = ",prob_dist)
    n_units = len(units)
    
    ## First get entropy of global system
    S_tot = -1*get_S(prob_dist)
#     print("Global Entropy Contribution = ",S_tot)

    spin_configs = all_configs(n)
    spin_df = pd.DataFrame(spin_configs) # df for global spin configs
    
    ## Now get entropy of units
    for unit in units:
        if len(unit) == 1:
            sign = -1.
        elif len(unit) == 3:
            sign = 1.
        else:
            print("ERROR - UNIT NOT A NODE OR LEAF")
            break
            
        configurations = all_configs(len(unit))  # gets possible values for intersection
        p_dist = []   # empty list to hold the probabilities over the different configurations
                
        ## Calculate entropy of marginal distribution
        for spin_config in configurations:
            index_list = []
            
            ## get index where original array matches spin_config
            for m in range(len(unit)):
                subsystem = unit[m]  # get the index of each spin
                index_list.append(spin_df.index[spin_df[subsystem]==spin_config[m]].tolist())
            final_indices = list(set(index_list[0]).intersection(*index_list))
            p_dist.append(np.sum(np.asarray(prob_dist)[final_indices]))
        S_tot = S_tot + sign*get_S(p_dist)  # sum entropy contribution from unit
        
    return(S_tot)