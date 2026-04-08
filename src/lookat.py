import numpy as np
from typing import Tuple
# This function determines the coordinate frame of the camera. Camera's coordinate frame is defned by the reference point (center
# of the camera) and the three orthonormal vectors xc, yc, zc that represents the  orthonormal basis of the coordinate system of
# the camera. The function calculates and returns the parameters that are needed in order to transform the 3D-coordinates of a 
# point given with respect to WCS (i.e the coordinate frame with O=[0,0,0] as reference point and the orthogonal axis x,y,z as
# axis of the coordinate system) into the 3D-coordinates of the point w.r.t the camera's frame. These parameters are the rotation
# matrix R (3 x 3 2D-array) which represents the transformation of the WCS' basis to the orthonormal basis of the camera's frame
# (i.e [xc, yc, zc]=R*[x, y, z] where x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1]) and the translation vector t (1 x 3 array) which
# represents the shift vector from the reference point O=[0,0,0] of the WCS to the reference point of the camera's coordinate 
# frame. The function takes as inputs the 3D-coordinates of the camera's center, the 3D-coordinates of the up vector of the
# camera and the 3D-coordinates of the target point that the camera is pointing towards. 
# Camera's coordinate frame can be represented by the camera's view matrix, which represents the coordinate frame transformation
# from WCS to camera's frame. Camera's view matrix is specified by a rotation matrix R that stands for the rotation transform of
# WCS' basis (x,y,z) into camera's coordinate system basis (xc,yc,zc) and a translation vector t from the O to reference point of
# camera's coordinate system. All the coordinates are given w.r.t WCS.
#
# INPUTS:
# eye: a vector of length 3 which represents the 3D-coordinates of the center of the camera w.r.t WCS.
# up: a vector of length 3 which represents the 3D-coordinates of the up vector of the camera w.r.t WCS.
# target: a vector of length 3 which represents the 3D-coordinates of the target point of the camera w.r.t WCS.
#
# OUTPUTS:
# R: a 3x3 2D-array which represents the rotation matrix that stands for the rotation transform of the orthogonal basis of WCS
#    to the basis of camera's coordinate system.
# t: a vector of length 3 that represents the 3D-coordinates of the shift vector of the WCS' center to the center of the camera
#    w.r.t WCS.


def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    # Calculate the camera's view matrix (i.e., its coordinate frame transformation specified
    # by a rotation matrix R, and a translation vector t).
    # :return a tuple containing the rotation matrix R (3 x 3) and a translation vector
    # t (1 x 3)
    
    # At first, the translation vactor has the same coordinates (w.r.t WCS) with the center of the camera (w.r.t WCS)
    t=eye
    R=np.zeros((3,3))
    
    # Compute the xc,yc,zc which are thw unit vectors of the orthonormal basis of the camera's coordinate system.
    CT=target-eye
    Euclidean_length_CT=np.linalg.norm(CT)
    # zc is the normalized CT
    zc=CT/Euclidean_length_CT

    # To compute yc, we need the vector k which is parallel to yc and it can be found if the projection of the up vector onto
    # zc is subtracted from the up vector.
    k=up - ( np.dot(up.T,zc)*zc )
    yc=k/np.linalg.norm(k)
    
    # The third vector xc of the basis is given by computing the cross product of the other two (yc,zc)
    xc=np.cross(yc,zc)

    # [xc, yc, zc]=R*[x, y, z] where x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1]. So R=[xc, yc, zc]
    R[:,0]=xc
    R[:,1]=yc
    R[:,2]=zc

    return [R,t]
