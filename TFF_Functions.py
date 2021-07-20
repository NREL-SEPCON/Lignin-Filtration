#This file contains the functions required to run a TFF module

#Import relevant packages/libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
idx = pd.IndexSlice
from scipy.optimize import least_squares
import time

#----------------------------------------------------------------------------------------------------------------- 
#Power law viscosity fit
def viscFit(data, m, b, n):
    return (m*data[0]**2+b)*data[1]**(n-1)
#-----------------------------------------------------------------------------------------------------------------
#This function takes the percent of cells to calculate before the rest are assumed constant, 
#the component input parameters, and the module geometry input parameters as inputs

#It iterates over the module and returns cellData, a dataframe with the output data for each cell

def TFF(comp_inputs, module):
    start = time.process_time()

    #Pre-allocate memory for cellData dataframe
    cellData = pd.DataFrame(data=np.zeros((module['N']+1,27)), index=np.arange(0,module['N']+1),columns=[
        [0,0,0,0,0,\
         1,1,1,1,1,\
         2,2,2,2,2,\
        3,3,3,3,3,3,3,3,3,3,3,3],\
        ['M','x_perm','x_ret','massbal','osmo_p',\
            'M','x_perm','x_ret','massbal','osmo_p',\
            'M','x_perm','x_ret','massbal','osmo_p',\
    'flux','len_tot','F_ret','theta','A','l','Pdrop','P_ret','visc','TMP','MassBaltot','osmo_p_tot']])


    #Set initial inputs for "cell 0"
    cellData.loc[idx[0], idx[:, 'x_ret']] = comp_inputs['x'].to_list() #Retentate mass fraction
    cellData.loc[idx[0], idx[:, 'x_perm']] = cellData.loc[idx[0], idx[:, 'x_ret']].values/10 #Permeate mass fraction
    cellData.loc[idx[0], idx[3, 'F_ret']] = comp_inputs['MFR'][0] #Retentate mass flow rate, kg/s
    cellData.loc[idx[0], idx[3, 'P_ret']] = module['P_in'] #Retentate/feed pressure, psi
    cellData.loc[idx[0], idx[3, 'l']] = module['length']/module['N'] #Module length, m
    cellData.loc[idx[0], idx[3, 'len_tot']] = module['length']/module['N']
    cellData.loc[idx[0], idx[3, 'A']] = module['SA']/module['N'] #Module area, m^2
    cellData.loc[idx[0],idx[:,'flux']] = module['F_p']/module['SA'] #Permeate flux, kg/m^2/s
    F_p = module['F_p']/module['N'] #Permeate mass flow rate, kg/s
    num_components=len(comp_inputs)

        #Solve for x_permeate values
    def obj(xp):
        xr = (cellData.loc[idx[j-1], idx[:, 'x_ret']].values-theta*xp)/(1-theta)
        LHS = xp*module['alpha']*(TMP-comp_inputs['vantHoff'][0]*module['osmo_factor']*(M[0]*xr[0]-xp[0])\
                                  -comp_inputs['vantHoff'][1]*module['osmo_factor']*(M[1]*xr[1]-xp[1])\
                                  -comp_inputs['vantHoff'][2]*module['osmo_factor']*(M[2]*xr[2]-xp[2]))
       
        RHS = (M*xr-xp)*(1-xp[0]-xp[1]-xp[2])
        diffList = [LHS[0]-RHS[0], LHS[1]-RHS[1], LHS[2]-RHS[2]]
        return diffList


    for j in range(1,module['N']+1):
        #Calculate retentate density and axial velocity
        c_ret = solveConc(np.array(cellData.loc[idx[j-1], idx[:, 'x_ret']].values)) #Solute concentrations in retentate, kg/m^3
        rho_ret = 1000+c_ret.sum() #Retentate density, kg/m^3
        v = cellData.loc[idx[j-1], idx[3, 'F_ret']]/(rho_ret) / module['Ac_tot'] #Axial velocity, m/s
        theta = F_p/cellData.loc[idx[j-1], idx[3, 'F_ret']]
        cellData.loc[idx[j], idx[3, 'theta']] = theta

        #Viscosity is estimated as a function of retentate side shear rate
        Q = cellData.loc[idx[j-1],idx[3,'F_ret']]/rho_ret #Channel flowrate, m^3/s
        shear_rate = 4*Q/(3.1415*(module['d_channel']/2)**3) #Shear rate at membrane surface, 1/s
        percent_vol_reduction = module['sysMRR']*100
        params = [0.00013123708177558568, 1.168890295612679, 0.9855078293755977] #viscosity parameters
        cellData.loc[idx[j-1], idx[3, 'visc']] = viscFit(np.array([[shear_rate],[percent_vol_reduction]]),*params)/1000

        #Calculate axial pressure drop, transmembrane pressure drop
        Re_N = (rho_ret)*v*module['d_channel']/cellData.loc[idx[j-1], idx[3, 'visc']] #Modified Reynolds number
        A = (2.457*np.log(1/((7/Re_N)**0.9+0.27*module['eta']/module['d_channel'])))**16 #Reynolds number parameter
        B = (37530/Re_N)**16 #Reynolds number parameter
        f = 2*((8/Re_N)**12 + 1/(A+B)**1.5)**(1/12) #Friction factor
        Pdrop = f*2*(rho_ret)*v**2/module['d_channel']*cellData.loc[idx[j-1],idx[3,'l']] #Axial pressure drop, Pa
        P_ret = cellData.loc[idx[j-1], idx[3, 'P_ret']] - Pdrop #Final retentate pressure
        TMP = (cellData.loc[idx[j-1], idx[3, 'P_ret']]+P_ret)/2 - module['P_perm'] #Transmembrane pressure drop, Pa
        cellData.loc[idx[j], idx[3, 'Pdrop']] = Pdrop
        cellData.loc[idx[j], idx[3, 'P_ret']] = P_ret
        cellData.loc[idx[j], idx[3, 'TMP']] = TMP

        #Calculate polarization concentration coefficient, M
        v_w = cellData.loc[idx[j-1],idx[3,'flux']]/(rho_ret) #Solute velocity, m/s
        M_const = (v_w**3*cellData.loc[idx[j-1],idx[3,'len_tot']]*module['d_channel']/2/4/v/comp_inputs['diff']**2)
        M_entr = 1.536*M_const**(1/3)+1
        M_far = M_const + 6 - 5*np.exp(-(M_const/3)**(1/2))
        M = M_entr #Concentration polarization coefficient
        for count in range(0,3):
            if M_far[count] < M_entr[count]:
                M[count] = M_far[count]
        cellData.loc[idx[j],idx[:,'M']] = M.values

        #Solve for permeate mass fraction values
        guess = cellData.loc[idx[j-1],idx[:,'x_perm']].values
        bounds = ([0,0,0],[cellData.loc[idx[j-1],idx[0,'x_ret']],cellData.loc[idx[j-1],idx[1,'x_ret']],cellData.loc[idx[j-1],idx[2,'x_ret']]])
        
        res = least_squares(obj,guess,bounds=bounds)
        x_perm = res.x
        cellData.loc[idx[j], idx[:, 'x_perm']] = x_perm

        #Solve for retentate flowrate, mass fractions
        F_ret = cellData.loc[idx[j-1], idx[3, 'F_ret']] - F_p
        x_ret = (cellData.loc[idx[j-1], idx[:, 'x_ret']].values-theta*x_perm)/(1-theta)
        cellData.loc[idx[j], idx[3, 'F_ret']] = F_ret
        cellData.loc[idx[j], idx[:, 'x_ret']] = x_ret            

        #Evaluate mass balances on total module and on each component
        massBaltot = cellData.loc[idx[j-1], idx[3, 'F_ret']] - F_ret - F_p
        massBal = cellData.loc[idx[j-1], idx[3, 'F_ret']]*cellData.loc[idx[j-1], idx[:, 'x_ret']].values -\
            F_ret*cellData.loc[idx[j], idx[:, 'x_ret']].values - \
            module['F_p']*cellData.loc[idx[j], idx[:, 'x_perm']].values
        cellData.loc[idx[j], idx[3, 'massBaltot']] = massBaltot
        cellData.loc[idx[j], idx[:, 'massbal']] = massBal


        #Osmotic Pressure
        osmo_p_tot = 0
        for counter in range (0,num_components):
            cellData.loc[idx[j],idx[counter,'osmo_p']] = comp_inputs['vantHoff'][counter]*module['osmo_factor']*(M[counter]*x_ret[counter] - x_perm[counter])
            osmo_p_tot += cellData.loc[idx[j],idx[counter,'osmo_p']]

        cellData.loc[idx[j], idx[3, 'osmo_p_tot']] = osmo_p_tot

        #Calculate cell length and area and flux
        cellData.loc[idx[j], idx[3, 'A']] = (F_p*(1-x_perm.sum()))/(module['free_perm']*(TMP - osmo_p_tot))  
        cellData.loc[idx[j],idx[3,'l']] = cellData.loc[idx[j], idx[3, 'A']]/module['circ_tot']
        cellData.loc[idx[j], idx[3, 'len_tot']] = cellData.loc[idx[j-1], idx[3, 'len_tot']] + cellData.loc[idx[j], idx[3, 'l']]
        cellData.loc[idx[j], idx[3, 'flux']] = F_p/cellData.loc[idx[j], idx[3, 'A']]

    return cellData

#-----------------------------------------------------------------------------------------------------------------
#moduleOutputsCalc takes the cellData dataframe and calculates the overall module output values
#It takes cellData as an input, calculates a variety of output parameters, and returns them in a dictionary called moduleOutputs

def moduleOutputsCalc(comp_inputs,module,cellData):
    
    F_p = module['F_p']/module['N']
    
    #Permeate: concentrations, mass fractions, total MFR    
    x_p_out= cellData.loc[idx[:],idx[:,'x_perm']].sum().values / module['N']
    c_p_out =  solveConc(x_p_out)
    F_p_out = F_p*module['N']
    recov_p = (F_p_out*x_p_out/(cellData.loc[idx[0],idx[:,'x_ret']]*cellData.loc[idx[0],idx[3,'F_ret']])).values
    density_p = 1000+c_p_out.sum()
    
    #Retentate: same as above
    x_r_out = cellData.loc[idx[module['N']],idx[:,'x_ret']].values
    c_r_out = solveConc(x_r_out)
    F_r_out = cellData.loc[idx[module['N']],idx[3,'F_ret']]
    recov_r = (F_r_out*x_r_out/(cellData.loc[idx[0],idx[:,'x_ret']]*cellData.loc[idx[0],idx[3,'F_ret']])).values
    pur_p = x_p_out[0]/x_p_out.sum()
    visc_ret = cellData.loc[idx[module['N']],idx[:,'visc']].values  
    density_r = 1000+c_r_out.sum()
    
    #Module: total length, total area, total axial Pdrop, TMP, Flux, mass balance, Power requirement
    A_total = cellData.loc[idx[:],idx[3,'A']].sum()
    l_calc = cellData.loc[idx[:],idx[3,'l']].sum()
    Pdrop_tot = cellData.loc[idx[:],idx[3,'Pdrop']].sum()
    TMP = (module['P_in'] + cellData.loc[idx[module['N']],idx[3,'P_ret']] - Pdrop_tot)/2 - module['P_perm']
    J_module = F_p_out/A_total
    massBal_overall = comp_inputs['MFR'][0] - F_p_out - cellData.loc[idx[module['N']],idx[3,'F_ret']].sum()
    massBal_comp = comp_inputs['MFR'][0]*cellData.loc[idx[0],idx[:,'x_ret']].values - F_p_out*x_p_out - cellData.loc[idx[module['N']],idx[3,'F_ret']]*cellData.loc[idx[module['N']],idx[:,'x_ret']].values
    Wp = Pdrop_tot / density_r
    eff=0.6
    P = Wp * comp_inputs['MFR'][0] / eff
    
    moduleOutputs = dict(density_p=density_p, x_perm=x_p_out,c_perm=c_p_out,F_perm=F_p_out,recov_perm=recov_p,density_r = density_r, x_ret=x_r_out,c_ret=c_r_out,F_ret=F_r_out,visc_ret=visc_ret,recov_ret=recov_r,A_tot=A_total,l_calc=l_calc,Pdrop_tot=Pdrop_tot,TMP=TMP,J_mod=J_module,massBal=massBal_overall,massBalcomp = massBal_comp,P=P)
    return moduleOutputs

#--------------------------------------------------------------------------------------------------------------
#Fit for pump calibration setting to kg/s
def mfr(pumpSetting,m,b):
    return m*pumpSetting+b

#--------------------------------------------------------------------------------------------------------------
#Updated flux decline fit where a and c params from above are based on initial and asymptotic flux values
#Only m needs to be calculated with fit
def fluxFit(vol, init, asy, m):
    flux = (init-asy)*vol**(-m)+asy
    if flux > init:
        flux = init
    return flux

def concObj(c, x0, x1, x2):
    c[0] = (c.sum()+1000)*x0
    c[1] = (c.sum()+1000)*x1
    c[2] = (c.sum()+1000)*x2
    return np.array([c[0]/(c.sum()+1000)-x0,c[1]/(c.sum()+1000)-x1,c[2]/(c.sum()+1000)-x2])

def solveConc(x):    
    guess = x*1000+1
    bounds = ([0,0,0],np.inf)
    res = least_squares(concObj,guess,bounds=bounds,args=(x[0],x[1],x[2])) #np.array(x)
    return res.x