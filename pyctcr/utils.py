import numpy as np

def hat(mat):
    return np.array([[0, -mat[2], mat[1]],
                     [mat[2], 0 , -mat[0]],
                     [-mat[1], mat[0], 0]])

def invhat(mat):
    return np.array([mat[2,1], mat[0,2], mat[1,0]])