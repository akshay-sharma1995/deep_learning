'''
Utility functions 
Free from to add functions if needed
'''

import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_airfoils(airfoil_x, airfoil_y, figname):
    '''
    plot airfoils: no need to modify 
    '''
    idx = 0
    fig, ax = plt.subplots(nrows=4, ncols=4)
    for row in ax:
        for col in row:
            col.scatter(airfoil_x, airfoil_y[idx, :], s=0.6, c='black')
            col.axis('off')
            col.axis('equal')
            idx += 1
    plt.savefig("./{}".format(figname))
    plt.show()
    plt.close()

def plot_prop(prop, prop_name, std=None):
    prop = np.array(prop)
    figure, ax = plt.subplots(1,1,figsize=(16,9))
    ax.plot(prop, color='orangered')
    ax.set(xlabel="epoch", ylabel=prop_name)

    if(std!=None):
        std = np.array(std)
        ax.fill_between(range(std.shape[0]), prop-std, prop+std, facecolor='peachpuff', alpha=0.7)
    plt.savefig("{}.png".format(prop_name))
    plt.close()

