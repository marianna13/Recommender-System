# Utilities


from scipy.sparse import csr_matrix
import numpy as np


def from_pd_to_array(data, user_col, item_col, rating_col):

  '''Converts pandas DataFrame to pivot table of ratings'''

  # create a sparse matrix of using the (rating, (rows, cols)) format
  rows = data[user_col].cat.codes
  cols = data[item_col].cat.codes
  
  rating = data[rating_col]
  ratings = csr_matrix((rating.astype(np.float32), (rows.astype(np.int64), cols.astype(np.int64))))
  ratings.eliminate_zeros()
  return np.array(ratings.todense(), dtype=np.float32)


def mse(A,B):

  '''Calculates mean squared error'''

  mse = (np.square(A - B)).mean()
  return mse