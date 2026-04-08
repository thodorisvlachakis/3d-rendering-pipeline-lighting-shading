import numpy as np
import math
# This class is the definition of the Transorm class, whose objects stand for affine transormations.
# This class definition is used in order to define the attributes and the methods needed to completely declare affine
# transormations.
# ATTRIBUTES: 
# mat: This is an instance attribute called mat, which is a 4×4 2D-array that stands for an affine transformation. It is initialized
#      as a 4×4 identity 2D-array.
#
# METHODS: Declare the interface of the class to perform or define an affine transormation.
# __init__(self): This function is the constructor of the class. It is called when a Transform object is created. It initializes
#                 the object by setting its instance attribute "mat" to be a 4×4 identity 2D-array.
#
# rotate(self, theta: float, u: np.ndarray)->None:
# This function calculates the rotation matrix, that corresponds to a clockwise rotation by an angle theta in rads (which is
# given as input of the funtion) about an axis whose direction is given by the unit vector u(given as input). This fuction updates
# the mat array (attribute) appropriately in order to include this rotation matrix in the affine transformation. The function has
# no output. 
#
# translate(self, t: np.ndarray)->None:
# This function takes as input a shift vector t, corresponding to the shift vector (offset) that respects to the affine transormation.
# The function updates the mat array (attribute) appropriately in order to include this shift vector in the affine transformation
# and has no output.
#
# transform_pts(self, pts: np.ndarray)->np.ndarray:
# This function takes as input a N×3 2D-array whose each row represents the 3D-coordinates of a point (N points in total). The
# function transforms these coordinates into new coordinates, according to the array mat that represents the transformation
# matrix of a declared affine transformation. The function calculates a N×3 2D-array whose its row represents the transformed
# 3D-coordinates of the respective point in the array pts (given as input). The fucntion returns this N×3 2D-array that includes
# the transormed coordinates. 


class Transform:
# Interface for performing affine transformations.

    def __init__(self):
        # This function initializes a Transform object.
        self.mat=np.identity(4)

    def rotate(self, theta: float, u: np.ndarray)->None:
        # This function calculates the rotation matrix which is declared for the affine transformation.
        # The rotation matrix R is a 3×3 2D-array that is given by Rodrigue's Rule:
        # R= (1-cos(theta) )*u*transpose(u) + cos(theta)*I + sin(theta)*Tu, where I is the 3×3 identity matrix and Tu represents
        # the outer pruduct of the vector u and the position vector corresponding to the point p to be transormed. Tu is directly
        # defined only given the vector u. 
        
        I=np.identity(3)
        # calculate u*transpose(u).
        U=np.outer(u,u)
        Tu=np.array([ [0,-u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0] ])
        R= ( (1-math.cos(theta))*U ) + ( math.cos(theta)*I ) + ( math.sin(theta)*Tu )

        # R is the rotation matrix. Update the transformation matrix mat that defines the affine transformation.
        self.mat[:3,:3]=R

        return None

    def translate(self, t: np.ndarray)->None:
        # This function updates the instance attribute mat with the given shift vector.
        self.mat[:3,3]=self.mat[:3,3]+t
    
    def transform_pts(self, pts: np.ndarray)->np.ndarray:
        # This function transforms the 3D-points given in the N×3 2D-array pts according to mat array. The transformation
        # refers to the transformation of the coordinates of the points.
        # !!!Very simple to transform the coordinates of some point according to an affine transormation, using the homogeneous
        # coordinates
        N=pts.shape[0]
        Homogeneous_pts=np.zeros((N,4))
        Homogeneous_pts[:,:3]=pts
        Homogeneous_pts[:,3]=1

        # Let T_pts will be a N×3 2D-array whose its row includes the 3D-coordinates of the affine transormation of the point in
        # the respective row in the pts. Let Hom_T_pts is a N×4 2D-array whose its row includes the homogeneous coordinates of the
        # point in the respective row in T_pts.
        Hom_T_pts=np.zeros((N,4))
        for i in range(N):
            Hom_T_pts[i,:]=np.dot( self.mat,Homogeneous_pts[i,:] )
        
        T_pts=Hom_T_pts[:,:3]
        return T_pts
