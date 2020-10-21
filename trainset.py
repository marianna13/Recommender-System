import numpy as np


class Trainset:

  def __init__(self,ratings, test_size=None):
    '''ratings:pivot table with ratings as values, type: numpy ndarray
    '''

    self.train_data = ratings.copy()
    self.n_items = ratings.shape[1]
    self.n_users = ratings.shape[0]
    self.test_size = test_size

    if test_size is not None:
      self.split()

  def all_ratings(self):

    if self.test_size is not None:
      self.split()

    for u in range(self.n_users):
      for i in range(self.n_items):

        if self.train_data[u,i]!=0:
          yield u,i,self.train_data[u,i]


  def get_testing_indices(self):
    '''randomly chooses items and users for further testing
    '''

    size = int(min(self.n_users,self.n_items)*self.test_size)

    self.ux = np.random.choice(np.arange(self.n_users),size)
    self.ix = np.random.choice(np.arange(self.n_items),size)
  
  def split(self):

    '''creates an  array without chosen (users,items)
    '''

    self.get_testing_indices()

    # to save original data for testing
    self.test_data = self.train_data.copy()

    # all the chosen elemnts in the training set must be zero
    self.train_data[self.ux,self.ix] = 0
