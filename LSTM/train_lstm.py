import numpy as np
import scipy as sp
import lstm
import matplotlib.pyplot as plt

def data8(n=100):
    # Generates '8' shaped data
    y = np.linspace(0,1, n)

    x = np.append(np.sin(2*np.pi*y), (-np.sin(2*np.pi*y)))

    return np.column_stack((x,np.append(y,y))).astype(dtype=np.float32)


def train_with_sgd(model,x,y,learning_rate,epochs,n_batch=100):
    for epoch in range(epochs):
        model.sgd_step(x,y,learning_rate)
        if epoch % n_batch==0:
            [predictions,cell]=model.predict(x)
            plt.scatter(predictions[0],predictions[1],'c')
            plt.show()
            print "Epoch: %s, Cost: %f" %(epoch,model.compute_cost(x,y))

data=data8()
lead_data=np.roll(data,1)
model=lstm.LSTM(2,2)
learning_rate=1e-3
epochs=10000
plt.scatter(data[:,0],data[:,1])
plt.show()


