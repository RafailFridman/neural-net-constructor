import numpy as np
from time import time
import dill as pickle
#from matplotlib import pyplot as plt
#from pandas import DataFrame
#from sklearn import datasets
#from collections import Counter
#from sklearn.datasets import make_moons
#from sklearn.datasets import make_circles

def sigmoid(x):
    #print(x)
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ReLu(x):
    return x * (x >= 0)

def compute_linear(data,W,B):
    return W.dot(data) + B

def quadraticcost(output, ylabels):
    m = ylabels.shape[1]  # number of examples
    cost = np.sum(1/(2*m)*(output-ylabels)**2)
    return cost

class NetModel:
    def __init__(self,hidden_layers,activations = dict(),input_layer=1,seed = 2):
        self.layers = hidden_layers
        self.activations = activations
        self.input_layer = input_layer
        self.seed = seed
        self.length = len(self.layers)

        np.random.seed(seed)
        self.parameters = dict()
        self.layers.insert(0, input_layer)
        self.der = {sigmoid: lambda x:sigmoid(x)*(1-sigmoid(x)),
                    tanh: lambda x:1-(tanh(x))**2,
                     quadraticcost : lambda a,y: a-y,
                     ReLu: lambda x: 1*(x>=0)}


        for l_num in range(1,len(self.layers)):
            self.parameters["w"+str(l_num)] = np.random.uniform(-1,1,size=(self.layers[l_num],self.layers[l_num-1]))*0.01 #dont forget to *0.01
            self.parameters["b"+str(l_num)] = np.zeros((self.layers[l_num],1))
            self.parameters["ac"+str(l_num)] = self.activations[l_num] if l_num in self.activations.keys() else None

    def forward_propagation(self,data):
        cur = data.transpose()   # cur features - rows; examples - columns;
        cache = dict()
        acs = self.activations
        cache["A0"]=cur
        for l in range(1,self.length+1): # length of net model - number of layers
            cache["Z"+str(l)]=compute_linear(cur,self.parameters["w"+str(l)],self.parameters["b"+str(l)])
            cache["A"+str(l)]=acs[l](cache["Z"+str(l)]) if acs[l] is not None else cache["Z"+str(l)] # applying activation if not None
            cur = cache["A"+str(l)]

        self.output = cache["A"+str(l)]
        self.cache = cache

    def back_propagation(self, ylabels, printq = False, cost= quadraticcost):
        changes = dict()
        ylabels=ylabels.transpose()
        m=len(self.output[0])
        dcura = self.der[cost](self.output,ylabels)
        dcurz = dcura*self.der[self.parameters['ac'+str(self.length)]](self.cache['Z'+str(self.length)])
        if printq==True:
            print(self.output)
        for l in range(self.length,0,-1):
            changes["dw"+str(l)] = 1/m*dcurz.dot(self.cache['A'+str(l-1)].transpose())+self.lambd/m*self.parameters["w"+str(l)]
            changes["db"+str(l)] = 1/m*np.sum(dcurz,axis=1,keepdims=True)
            if l==1:
                break
            dcura = (self.parameters['w'+str(l)].transpose()).dot(dcurz)
            dcurz = dcura*self.der[self.parameters['ac'+str(l-1)]](self.cache['Z'+str(l-1)])
        self.changes = changes

    def l2_cost(self,output, ylabels):
        m = ylabels.shape[1]  # number of examples
        l2 = self.lambd/( m * 2 )*np.sum([np.sum(np.square(self.parameters["w"+str(i)])) for i in range(1,self.length)])
        cost = np.sum(1/(2*m)*(output-ylabels)**2)
        #print(cost,l2)
        return cost + l2

    def update_weights(self,learning_rate=0.05):
        for l in range(1,self.length+1):
            self.parameters["w"+str(l)]=self.parameters["w"+str(l)]-learning_rate*self.changes["dw"+str(l)]
            self.parameters["b"+str(l)]=self.parameters["b"+str(l)]-learning_rate*self.changes["db"+str(l)]

    def net_train(self,data,ylabels, iterations = 1000,lr = 0.05, lambd = 0, printq = 1000):
        try:
            self.lambd = lambd
            for i in range(iterations):
                self.forward_propagation(data)
                self.back_propagation(ylabels)
                self.update_weights(learning_rate=lr)
                if i % printq== 0:
                    if self.lambd==0:
                        print("Cost after iteration %i: %f" %(i, quadraticcost(self.output.transpose(),ylabels)))
                    else:
                        print("Cost after iteration %i: %f" %(i, self.l2_cost(self.output.transpose(),ylabels)))
        except KeyboardInterrupt:
            print("KeyboardInTerrupt")
            return self.parameters
        return self.parameters

    def SGD(self, data, ylabels, iterations = 1000, batch_size = 16, lr = 0.05, printq = 1000,seed = 3,lambd = 0):
        try:
            self.lambd = lambd
            np.random.seed(seed)
            train = list(zip(data,ylabels))
            n = len(train)
            for i in range(iterations):
                np.random.shuffle(train)
                batches = [train[k:k+batch_size] for k in range(0, n, batch_size)]
                for batch in batches:
                    batch_data = np.array([i[0] for i in batch])
                    batch_ylabels = np.array([i[1] for i in batch])

                    self.forward_propagation(batch_data)
                    self.back_propagation(batch_ylabels)
                    self.update_weights(learning_rate=lr)

                if i % printq== 0:
                    self.forward_propagation(data)
                    if self.lambd==0:
                        print("Cost after iteration %i: %f" %(i, quadraticcost(self.output.transpose(),ylabels)))
                    else:
                        print("Cost after iteration %i: %f" %(i, self.l2_cost(self.output.transpose(),ylabels)))
        except KeyboardInterrupt:
            print("KeyboardInTerrupt")
            return self.parameters
        return self.parameters

    def fit(self, data, ylabels, method = 'SGD', iterations = 10000, lr = 0.05, printq = 1000, batch_size = 16, seed = 3, lambd = 0):
        if method == 'SGD':
            self.SGD(data, ylabels, iterations=iterations, lr=lr, batch_size=batch_size, printq=printq, seed = seed, lambd = lambd)
        elif method == 'GD':
             self.net_train(data, ylabels, iterations=iterations, lr=lr, printq=printq, lambd = lambd)

    def predict(self,data,printq = False):
        self.forward_propagation(data)
        if printq:
            print(self.output.transpose())
        return np.round(self.output).transpose()

    def save(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self.parameters, f)

    def load(self,path):
        with open(path, 'rb') as f:
            self.parameters = pickle.load(f)

    def __repr__(self):
        return str(self.parameters)


#additional functions
def plot_labels(data,target):
    df = DataFrame(dict(x=data[:,0], y=data[:,1], label=target))
    colors = {0:'red', 1:'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()

def f_predict(data,trained_net,printq=False):
    if printq:
        print(f_fp(data,trained_net)[0].transpose())
    return np.round(f_fp(data,trained_net)[0]).transpose()

def f_fp(data,nmodel): # data features - columns; examples - rows;
    cur = data.transpose()   # cur features - rows; examples - columns;
    cache=dict()
    acs={key:value for key,value in nmodel.items() if key.startswith("ac")} # getting activation functions from parameters
    cache["A0"]=cur
    for l in range(1,len(nmodel)//3+1): # length of net model - number of layers
        cache["Z"+str(l)]=compute_linear(cur,nmodel["w"+str(l)],nmodel["b"+str(l)])
        cache["A"+str(l)]=acs["ac"+str(l)](cache["Z"+str(l)]) if acs["ac"+str(l)]!=None else cache["Z"+str(l)] # applying activation if not None
        cur = cache["A"+str(l)]
    output = cache["A"+str(l)]
    return output, cache

def num(n):
    cls=np.zeros(10)
    cls[n]=1
    return cls

def fromnum(num):
    d=np.zeros(10)
    d[num]=1.
    return d

def evaluate_numbers(data,ylabel,trained_net,printq=False):
    return np.all(predict(np.array([data]),trained_net)==np.array(ylabel))

def evaluate_clothes(img,label,net):
    if np.argmax(predict(img.reshape(1,784),net,printq=False))==label:
        return True
    return False

def loadmnist(path):
    mnist_raw = loadmat(path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    return mnist

if __name__ == "__main__":
    net = NetModel([3,1],activations={1:ReLu,2:sigmoid},input_layer=2,seed=4)
    np.random.seed(3)
    data=np.random.randn(20,10)
    ylabels=np.random.randn(20,1)
    #net.back_propagation(ylabels)
    np.random.seed(3)
    net.fit(data,ylabels,method='SGD',batch_size=16, iterations=6000,printq=1000)
    #print(time()-st)
    #print("{")

