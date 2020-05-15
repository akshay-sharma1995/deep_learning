import numpy as np
from argparse import parser

def sigma(x):
    return 1.0/(1.0+np.exp(-x))

def del_g(x):
    return sigma(1.702*x) * (1 + x*(1-sigma(1.702*x))*1.702)

def g(x):
    return x*sigma(1.702*x)

def gd_steps(x0,lr,n,use_momentum=False,beta=0.0):
    if momentum==False:
        x = x0
        for i in range(n):
            x_new = x - lr*del_g(x)
            print("x_{}: {}, g(x): {}".format(i+1,x_new,g(x_new)))
            x = x_new

    else:
        x = x0
        v = del_g(x0)
        for i in range(n):
            v_new = beta*v + (1-beta)*del_g(x)
            x_new = x - lr*v_new
            print("x_{}: {}, v(x): {},  g(x): {}".format(i+1,x_new,v_new,g(x_new)))
            x = x_new
            v = v_new

