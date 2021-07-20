#Import statements
import numpy as np
import pandas as pd
idx = pd.IndexSlice

#This function returns a module object with all relevant settings for the 200 Da membrane
def setMod(P_in, N_in, flux_init, osmo_factor):
    
    #Set system inputs
    free_perm =  (6.222842366)/3600/100000/2 #Free water permeance, kg/m^2/s/Pa (LMH/bar)
    alpha = np.array([6.86885601e-07, 9.86009340e-06, 8.94422103e-05 ]) #Membrane selectivity (Ksolv/Ksolute), 1/Pa
    
    d_channel = 0.0070104 #Membrane channel diameter, m
    num_channels = 1 #Number of channels in module
    length = 0.357 #Length of module, m
    eta = 1*10**(-5) #Roughness of membrane surface, m
    P_in = (P_in)*6894.75729 #Inlet pressure, Pa (psi)
    P_perm = (14.6959)*6894.75729 #Permeate side pressure - assumed constant atmospheric, Pa (psi)
    N = N_in #Number of cross-flow cells in module (for calculation loop)
    SA=np.pi*d_channel*length*num_channels #Membrane surface area, m^2
    F_p = flux_init*SA #Permeate mass flow rate, kg/s
    sysMRR = 1E-10
    
    #Put all needed system inputs into dictionary
    module = dict(sysMRR = sysMRR,F_p=F_p, alpha=alpha, SA=SA, length=length, d_channel=d_channel, free_perm=free_perm,Ac_tot=np.pi*(d_channel/2)**2*num_channels,circ_tot=np.pi*d_channel*num_channels,eta=eta,P_in=P_in,P_perm=P_perm,N=N,osmo_factor=osmo_factor)

    return module