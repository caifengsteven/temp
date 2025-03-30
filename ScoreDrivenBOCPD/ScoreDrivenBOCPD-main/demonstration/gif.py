# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:11:36 2024

@author: Yvonne
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio
from ScoreDrivenBOCPD.sd_bocpd import SDBocpd,Hazard
from ScoreDrivenBOCPD.prob_model import GaussianModel

def dynamics(y_t1, T, t, dyn = "sin"):
    
    if dyn == "sin":
        f_t = 0.5* np.sin(t/30)
  
    if dyn == "step":
        if (t < int(T/4)):
            f_t = 0.5
        if (t >= int(T/4)):
            f_t = -0.5
        if (t >= int(T/2)):
            f_t = 0.5
        if (t >= int(3*T/4)):
            f_t = -0.5
    
    return  f_t


def sim_data(T,p,rho):
    data = []
    cps = []
    mean = np.random.normal(mean0, var0)
    data.append(np.random.normal(mean,var))
    
    for t in range(1,T): 
        
        if np.random.random() < p:
            
            mean = np.random.normal(mean0, var0)
            cps.append(t)
            data.append(np.random.normal(mean,var))
            
            continue
        
        data.append(np.random.normal(mean+rho*(data[t-1]-mean),var*(1-rho**2)))

    return data,cps



def plot_for_offset(data, cpss, T):

    fig, axes = plt.subplots(1,1,figsize=(10,5))
    ax1 = axes
    
    ax1.scatter(range(1,len(data)+1),data)
    ax1.set_xlim(0,T)
    for cp in cpss:
        if cp<len(data):
            ax1.axvline(cp, c='red', ls='dotted',lw=2)

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


if __name__ == '__main__':

    T = 200
    q = 0
    d = 0
    p = 1/100
    var = 1
    init_cor = 0.2
    parameters = [0, 0.1, 0.9, 0.2, 1]
    init_var = 1
    mean0 = 0
    var0 = 1
    data,cps = sim_data(T,p,init_cor)
    
    model  = GaussianModel(mean0, var0, var, init_cor, var,parameters,q)
    
    SDB = SDBocpd(T, d, q)
    hazard = Hazard(T,1/100)
    R,cp = SDB.bocd(data,model,hazard)
    imageio.mimsave('./run_length_posterior.gif', [plot_for_offset(data[:i+1], cps, T) for i in range(100)], fps=10)




