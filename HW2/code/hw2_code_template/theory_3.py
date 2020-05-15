import numpy as np
import ipdb
def sigmoid(x):
    return np.divide(1.0,1.0 + np.exp(-x))

def tanh(x):
    expo = np.exp(2*x)
    return (expo - 1) / (expo + 1)

class lstm():
    def __init__(self, h0, c0, Wf, Wi, Wc, Wo, Uf, Ui, Uc, Uo, bf, bi, bc, bo):
        self.h = h0
        self.c = c0
        self.Wf = Wf
        self.Wi = Wi
        self.Wc = Wc
        self.Wo = Wo
        self.Uf = Uf
        self.Ui = Ui
        self.Uc = Uc
        self.Uo = Uo
        self.bf = bf
        self.bi = bi
        self.bc = bc
        self.bo = bo

    def step(self,x):
        f = sigmoid(np.matmul(self.Wf,x) + np.matmul(self.Uf, self.h) + self.bf)
        i = sigmoid(np.matmul(self.Wi,x) + np.matmul(self.Ui, self.h) + self.bi)
        c_ = tanh(np.matmul(self.Wc,x) + np.matmul(self.Uc, self.h) + self.bc)
         
        self.c = np.multiply(f, self.c) + np.multiply(i, c_)
        
        o = sigmoid(np.matmul(self.Wo,x) + np.matmul(self.Uo, self.h) + self.bo)

        self.h = np.multiply(o, tanh(self.c))
        print("f: {}".format(f))
        print("i: {}".format(i))
        print("c_: {}".format(c_))
        print("c: {}".format(self.c))
        print("o: {}".format(o))
        print("h: {}".format(self.h))

    def process_seq(self,X):
         
        # seq length
        n = X.shape[0]
        print("initial") 
        print("h: {}".format(self.h))
        print("c: {}".format(self.c))
        for i in range(n):
            print("time_step: {}".format(i+1))
            x = X[i]
            self.step(x)
            # print("h: {}".format(self.h))
            # print("c: {}".format(self.c))
            print("------------------------------------------------------------------------------------------------------------------")


def main():
    Wf = np.array([1,2]).reshape(1,-1)
    Uf = np.array([0.5]).reshape(1,-1)
    bf = np.array([0.2]).reshape(1,1)
    
    print("Wf: {}".format(Wf))
    print("Uf: {}".format(Uf))
    print("bf: {}".format(bf))

    Wi = np.array([-1,0]).reshape(1,-1)
    Ui = np.array([2]).reshape(1,-1)
    bi = np.array([-0.1]).reshape(1,1)

    print("Wi: {}".format(Wi))
    print("Ui: {}".format(Ui))
    print("bi: {}".format(bi))
    
    Wc = np.array([1,2]).reshape(1,-1)
    Uc = np.array([1.5]).reshape(1,-1)
    bc = np.array([0.5]).reshape(1,1)
    
    print("Wc: {}".format(Wc))
    print("Uc: {}".format(Uc))
    print("bc: {}".format(bc))
    
    Wo = np.array([3,0]).reshape(1,-1)
    Uo = np.array([-1]).reshape(1,-1)
    bo = np.array([0.8]).reshape(1,1)
    
    print("Wo: {}".format(Wo))
    print("Uo: {}".format(Uo))
    print("bo: {}".format(bo))
    
    h0 = np.zeros(shape=(1,1))
    c0 = np.zeros(shape=(1,1))
    
    X = np.array([[1,0],[0.5,-1]])
    X = np.expand_dims(X,2)

    seq_model = lstm(h0, c0, Wf, Wi, Wc, Wo, Uf, Ui, Uc, Uo, bf, bi, bc, bo) 

    seq_model.process_seq(X)

main()


