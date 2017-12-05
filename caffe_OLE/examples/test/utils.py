import numpy as np

def sample_points(N,C,D):
  # sample N points from 3 subspaces

  assert D==3, 'D must be 3 to sample 3d points'
  assert C==3, 'C must be 3 to sample 3d points'

  p1 = np.array([1,-1,3]);
  p2 = np.array([2,3,4]);
  p3 = np.array([-5,6,7]);

  # x = np.concatenate((np.ones((1,N/2)), 2*np.ones((1,N/2))),axis=1) 
  # y = np.concatenate((2*np.ones((1,N/2)), np.ones((1,N/2))),axis=1) 

  
  np.random.seed(1)
  x = np.random.uniform(size=(1,N))

  np.random.seed(42)
  y = np.random.uniform(size=(1,N))



  R = np.array([[ 0.707106781186548, 0.707106781186547, 0],[ -0.707106781186547, 0.707106781186548, 0],[ 0, 0, 1]])
  R2 = np.array( [[0.707106781186548, 0, 0.707106781186547],[ 0, 1, 0],[ -0.707106781186547, 0, 0.707106781186548]])

  normal = np.cross(p1 - p2, p1 - p3);

  
  d = p1[0]*normal[0] + p1[1]*normal[1] + p1[2]*normal[2];
  d = -d;

  z = (-d - (normal[0]*x) - (normal[1]*y))/normal[2];

  X1 = np.concatenate((x, y, z),axis=0);

  X1 = X1 - np.mean(X1, axis=1, keepdims=True)

  
  normal = np.cross(p1 - p2, p1 - p3).dot(R);
  d = p1[0]*normal[0] + p1[1]*normal[1] + p1[2]*normal[2];
  d = -d;
  z = (-d - (normal[0]*x) - (normal[1]*y))/normal[2];
  X2 = np.concatenate((x, y, z),axis=0);
  X2 = X2-np.mean(X2,axis=1, keepdims=True)
  
  normal = np.cross(p1 - p2, p1 - p3).dot(R2);
  d = p1[0]*normal[0] + p1[1]*normal[1] + p1[2]*normal[2];
  d = -d;
  z = (-d - (normal[0]*x) - (normal[1]*y))/normal[2]; 
  X3 = np.concatenate((x, y, z),axis=0);
  X3 = X3-np.mean(X3,axis=1, keepdims=True)
  

  return np.concatenate((X1,X2,X3), axis=1).T
