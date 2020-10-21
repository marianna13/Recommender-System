import numpy as np
import matplotlib.pyplot as plt
from trainset import Trainset
from utils import mse


class Recommender:

  def __init__(self, lr=0.005,reg=0.02,n_epochs=10, n_factors=50):
    self.lr = lr
    self.reg = reg
    self.n_epochs = n_epochs
    self.n_factors = n_factors

  def fit(self,ratings, test_size=None, verbose=0, biased=1):

    ts = Trainset(ratings,test_size=test_size)

    self.r_train = ts.train_data

    if test_size is not None:
      self.r_test = ts.test_data
      self.ux = ts.ux
      self.ix = ts.ix
    
    lr = self.lr
    reg = self.reg
    n_epochs = self.n_epochs
    n_factors = self.n_factors

  
    mu = 0 # mean
    s = 0.1 # standard deviation

    

    n_users = ts.n_users
    n_items = ts.n_items

    # intialization of biases bu and bi
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)

    # random intialization of pu and qi
    pu = np.random.normal(loc=mu,scale=s,size=(n_users,n_factors))
    qi = np.random.normal(loc=mu,scale=s,size=(n_items,n_factors))
    
    # for graphing
    self.loss = []

    # training loop
    for ep in range(n_epochs):

      if verbose:
        print('Proesssing epoch: {}'.format(ep+1))

      for u,i,r in ts.all_ratings():
        # compute dot product
        dot = 0
        for f in range(n_factors):
          dot += qi[i,f]*pu[u,f]
      
        # compute error
        e = r - dot - bu[u] -bi[i]

        # update biases
        if biased:
          bu[u]+=lr*(e-reg*bu[u])
          bi[i]+=lr*(e-reg*bi[i])

        # update factors
        for f in range(n_factors):

          pu[u,f]+=lr*(e*qi[i,f]-reg*pu[u,f])
          qi[i,f]+=lr*(e*pu[u,f]-reg*qi[i,f])
      
      self.loss.append(e)

      if verbose:
        print('Training loss: {}'.format(e))

      self.pu = pu
      self.qi = qi
      self.bu = bu
      self.bi = bi

  def graph_loss(self):

    plt.plot(np.arange(self.n_epochs), self.loss)

  def predict(self,u,i):
    
    pred = np.dot(self.qi[i],self.pu[u])+self.bu[u]+self.bi[i]
    return pred
  
  def validate(self):
    
    test_data = self.r_test
    data = self.r_train
 

    for u in self.ux:
      for i in self.ix:
        data[u,i] = self.predict(u,i)

    print('Mean Squared Error: {}'.format(mse(data,test_data)))
    
  
  

      