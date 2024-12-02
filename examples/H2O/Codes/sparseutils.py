from scipy.sparse import coo_matrix
import numpy as np

def as_coo_tuple(a):
  ms = coo_matrix(a)
  #print ('nnz', ms.nnz / np.prod(a.shape))
  return (ms.data,(ms.row, ms.col), ms.shape)

def from_coo_tuple(coo_tuple):
  if hasattr(coo_tuple, "shape"):
    return coo_tuple # not a COO tuple actually
  data, (i,j), shape = coo_tuple
  ms = coo_matrix((data, (i,j)), shape=shape)
  return ms.toarray()

