# Do not import any additional 3rd party external libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import pdb

class Activation(object):

    """
    Interface for activation functions (non-linearities).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    x: (batch_size, dim_in)
    forward: (batch_size, dim_in)
    derivative: (batch_size, dim_in)
    """

    # This class is a gimme as it is already implemented for you as an example (do not change)

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    x: (batch_size, dim_in)
    forward: (batch_size, dim_in)
    derivative: (batch_size, dim_in)
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # hint: save the useful data for back propagation
        self.state = np.divide(1.0,(1+np.exp(-x)))
        return self.state

    def derivative(self):
        return np.multiply(self.state,(1-self.state))


class Tanh(Activation):

    """
    Tanh non-linearity

    x: (batch_size, dim_in)
    forward: (batch_size, dim_in)
    derivative: (batch_size, dim_in)
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        exp_2x = np.exp(2*x)
        self.state = np.divide((exp_2x - 1),(exp_2x + 1))
        return self.state

    def derivative(self):
        return 1.0 - np.power(self.state,2)


class ReLU(Activation):

    """
    ReLU non-linearity
    x: (batch_size, dim_in)
    forward: (batch_size, dim_in)
    derivative: (batch_size, dim_in)
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.maximum(0,x)    
        return self.state

    def derivative(self):
        return 1.0*(self.state>0)

class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        # you can add variables if needed
        self.probs = None
        self.loss = 0.0
    
    def calc_probs(self,x):
        
        
        expo = np.exp(x - np.max(x,1).reshape(-1,1))
        sum_expo = np.sum(expo,axis=1).reshape(-1,1)
        
        self.probs = np.divide(expo,sum_expo)
        
        return self.probs

    def forward(self, x, y):
        """
        x: (batch, dim_in) --> input logits to the softmax function
        y: (batch,c) --> one hot labels with class size c
        derivative: (batch, dim_in)
        """

        # batch_size = y.shape[0]
        # c = y.shape[1]
        # self.one_hot_labels = y
        # self.logits = x
        
        # expo = np.exp(x - np.max(x,1).reshape(-1,1))
        # sum_expo = np.sum(expo,axis=1).reshape(-1,1)
        # self.probs = np.divide(expo,sum_expo)
        eps = 1e-10
        self.logits = x
        self.one_hot_labels = y
        labels = y.argmax(axis=1).reshape(-1,1)
        probs_to_use = (np.array([self.probs[i,labels[i]]+eps for i in range(labels.shape[0])]))
        
        self.loss = -1.0*np.log(probs_to_use).mean()
         
        return self.loss

    def derivative(self):
        return self.probs - self.one_hot_labels

# randomly intialize the weight matrix with dimension d0 x d1 via Normal distribution
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0,d1)


# initialize a d-dimensional bias vector with all zeros
def zeros_bias_init(d):
    return np.zeros(shape=(d,))


class MLP(object):

    """
    A simple multilayer perceptron
    (feel free to add class functions if needed)
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr):

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes
        self.nn_dim = [input_size] + hiddens + [output_size]
        # list containing Weight matrices of each layer, each should be a np.array
        self.W = [weight_init_fn(self.nn_dim[i], self.nn_dim[i+1]) for i in range(self.nlayers)]
        # list containing derivative of Weight matrices of each layer, each should be a np.array
        self.dW = [np.zeros_like(weight) for weight in self.W]
        # list containing bias vector of each layer, each should be a np.array
        self.b = [bias_init_fn(self.nn_dim[i+1]) for i in range(self.nlayers)]
        # list containing derivative of bias vector of each layer, each should be a np.array
        self.db = [np.zeros_like(bias) for bias in self.b]

        # You can add more variables if needed
        self.Z = [None]*self.nlayers
        self.Y = [None]*self.nlayers
        
        self.X = [None]

    def forward(self, x):
        """
        x : (d_in, batch)
        self.Z[-1] : (d_out, batch)
        """
        self.X = x
        self.Z[0] = np.matmul(self.X, self.W[0]) + self.b[0]

        for i in range(0,self.nlayers-1):
            self.Y[i] = self.activations[i](self.Z[i])
            self.Z[i+1] = np.matmul(self.Y[i], self.W[i+1]) + self.b[i+1]
        
        self.Y[-1] = self.criterion.calc_probs(self.Z[-1])

    def zero_grads(self):
        # set dW and db to be zero
        for i in range(len(self.dW)):
            self.dW[i] = 0*self.dW[i]
            self.db[i] = 0*self.db[i]
    

    def step(self):     
        # update the W and b on each layer
        for i in range(len(self.W)):
            self.W[i] -= self.lr*self.dW[i]
            self.b[i] -= self.lr*self.db[i]
            

    def backward(self, labels):
        batch_size = labels.shape[0]
        if self.train_mode:
            # calculate dW and db only under training mode
            grad_E_Z = self.criterion.derivative() 
            for i in range(self.nlayers-1,0, -1):

                self.dW[i] += (1.0/batch_size) * np.matmul(self.Y[i-1].T , grad_E_Z)
                self.db[i] += np.mean(grad_E_Z,0)
                grad_E_Y = np.matmul(grad_E_Z, self.W[i].T)
                grad_Y_Z = self.activations[i-1].derivative() 
                grad_E_Z = np.multiply(grad_E_Y, grad_Y_Z)
           
            self.dW[0] += np.matmul(self.X.T, grad_E_Z)
            self.db[0] += np.mean(grad_E_Z,0)

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        # training mode
        self.train_mode = True

    def eval(self):
        # evaluation mode
        self.train_mode = False

    def get_loss(self, labels):
        # return the current loss value given labels
        self.loss = self.criterion.forward(self.Z[-1], labels)
        return self.loss

    def get_error(self, labels):
        # return the number of incorrect preidctions gievn labels
        preds = np.argmax(self.criterion.probs,axis=1)
        error = np.sum(preds!=np.argmax(labels,axis=1)) / labels.shape[0]
        # pdb.set_trace()
        return error

    def save_model(self, path='p1_model.npz'):
        # save the parameters of MLP (do not change)
        np.savez(path, self.W, self.b)


# Don't change this function
def get_training_stats(mlp, dset, nepochs, batch_size):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))
    

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    for e in range(nepochs):
        print("epoch: ", e)
        train_loss = 0
        train_error = 0
        val_loss = 0
        val_error = 0
        num_train = len(trainx)
        num_val = len(valx)
        
        perm = np.random.permutation(len(trainx))
        trainx = trainx[perm]
        trainy = trainy[perm]
        for b in range(0, num_train, batch_size):
            mlp.train()
            mlp(trainx[b:b+batch_size])
            # mlp.backward(trainy[b:b+batch_size])
            # mlp.step()
            train_loss += mlp.get_loss(trainy[b:b+batch_size])
            train_error += mlp.get_error(trainy[b:b+batch_size])
            mlp.zero_grads()
            mlp.backward(trainy[b:b+batch_size])
            mlp.step()
        # training_losses += [train_loss/num_train]
        # training_errors += [train_error/num_train]
        # print("training loss: ", train_loss/num_train)
        # print("training error: ", train_error/num_train)
        training_losses += [train_loss/(num_train//batch_size)]
        training_errors += [train_error/(num_train//batch_size)]
        print("training loss: ", train_loss/(num_train//batch_size))
        print("training error: ", train_error/(num_train//batch_size))
        
        for b in range(0, num_val, batch_size):
            mlp.eval()
            mlp(valx[b:b+batch_size])
            val_loss += mlp.get_loss(valy[b:b+batch_size])
            val_error += mlp.get_error(valy[b:b+batch_size])
        validation_losses += [val_loss/(num_val//batch_size)]
        validation_errors += [val_error/(num_val//batch_size)]
        print("validation loss: ", val_loss/(num_val//batch_size))
        print("validation error: ", val_error/(num_val//batch_size))

    test_loss = 0
    test_error = 0
    num_test = len(testx)
    for b in range(0, num_test, batch_size):
        mlp.eval()
        mlp(testx[b:b+batch_size])
        test_loss += mlp.get_loss(testy[b:b+batch_size])
        test_error += mlp.get_error(testy[b:b+batch_size])
    test_loss /= (num_test//batch_size)
    test_error /= (num_test//batch_size)
    print("test loss: ", test_loss)
    print("test error: ", test_error)

    return (training_losses, training_errors, validation_losses, validation_errors)


# get ont hot key encoding of the label (no need to change this function)
def get_one_hot(in_array, one_hot_dim):
    dim = in_array.shape[0]
    out_array = np.zeros((dim, one_hot_dim))
    for i in range(dim):
        idx = int(in_array[i])
        out_array[i, idx] = 1
    return out_array


def main():
    # load the mnist dataset from csv files
    image_size = 28 # width and length of mnist image
    num_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    train_data = np.loadtxt("../dataset/mnist_train.csv", delimiter=",")
    test_data = np.loadtxt("../dataset/mnist_test.csv", delimiter=",") 

    # rescale image from 0-255 to 0-1
    fac = 1.0 / 255
    train_imgs = np.asfarray(train_data[:50000, 1:]) * fac
    val_imgs = np.asfarray(train_data[50000:, 1:]) * fac
    test_imgs = np.asfarray(test_data[:, 1:]) * fac
    train_labels = np.asfarray(train_data[:50000, :1])
    val_labels = np.asfarray(train_data[50000:, :1])
    test_labels = np.asfarray(test_data[:, :1])
    # convert labels to one-hot-key encoding
    train_labels = get_one_hot(train_labels, num_labels)
    val_labels = get_one_hot(val_labels, num_labels)
    test_labels = get_one_hot(test_labels, num_labels)

    print(train_imgs.shape)
    print(train_labels.shape)
    print(val_imgs.shape)
    print(val_labels.shape)
    print(test_imgs.shape)
    print(test_labels.shape)

    dataset = [
        [train_imgs, train_labels],
        [val_imgs, val_labels],
        [test_imgs, test_labels]
    ]

    # These are only examples of parameters you can start with
    # you can tune these parameters to improve the performance of your MLP
    # this is the only part you need to change in main() function
    hiddens = [200,80]
    # activations = [ReLU(), ReLU(), ReLU(), ReLU()]
    # activations = [ReLU(), ReLU()]
    activations = [Sigmoid(),Sigmoid()]
    lr = 0.001
    num_epochs = 50
    batch_size = 32

    # build your MLP model
    mlp = MLP(
        input_size=image_pixels, 
        output_size=num_labels, 
        hiddens=hiddens, 
        activations=activations, 
        weight_init_fn=random_normal_weight_init, 
        bias_init_fn=zeros_bias_init, 
        criterion=SoftmaxCrossEntropy(), 
        lr=lr
    )

    # train the neural network
    losses = get_training_stats(mlp, dataset, num_epochs, batch_size)

    # save the parameters
    mlp.save_model()

    # visualize the training and validation loss with epochs
    training_losses, training_errors, validation_losses, validation_errors = losses

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9))

    ax1.plot(training_losses, color='blue', label="training")
    ax1.plot(validation_losses, color='red', label='validation')
    ax1.set_title('Loss during training')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(training_errors, color='blue', label="training")
    ax2.plot(validation_errors, color='red', label="validation")
    ax2.set_title('Error during training')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('error')
    ax2.legend()
    
    plt.savefig("p1.png")
    # plt.show()


if __name__ == "__main__":
    main()
