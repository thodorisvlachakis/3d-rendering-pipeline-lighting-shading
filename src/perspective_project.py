import numpy as np
from typing import Tuple
from world2view import *
# This function calculates the 2D coordinates of the perspective porjections of given (specified) 3D points, on the image plane.
# Given the 3D coordinates of set of 3D points (i.e points in 3D space) w.r.t camrea's coordinate system, which is described by
# a rotation matrix R (which describes the rotation transform of the WCS to the camera's frame) and a shift vector t (which
# represents the shift of the WCS' reference point, i.e [0,0,0], to the reference point of the camera's coordinate system, i.e 
# the center of the camera), this function uses a pinhole (i.e the camera lens is a pinhole located at the center of the camera)
# perspective projecton model in order to compute the 2D coordinates on the camera plane (like CCD of the camera) that respect
# to each given 3D point. In order to use this perspective projection model, the focal length is also needed as input parameter.

# INPUTS:
# pts: a 3×N 2D-array whose each column includes the 3D-coordinates of a point (one of the initial specified points) with respect
#      to an initial coordinate system.
# focal: a float number which represents the distance of the camera's CCD from the center of the camera (center of camera lens).
#        This is measured in units used by camera's coordinate system.
# R: a 3×3 2D-array which represents the rotation matrix that stands for the rotation transform of the initial coordinate frame,
#    i.e the coordinate frame to which the coordinates of the specified points respect, to the coordinate frame of the camera.
# t: a vector of length 3 that represents the coordinates of the shift (translate) vector to the camera's coordinate frame. In
#    other words, this vector is the shift vector from the reference point of the initial coordinate frame to the reference point
#    of the new (i.e camera's frame) coordinate frame.
#
# OUTPUTS:
# Transfrormed_pts: a N×2 2D-array whose i-th row includes the 2D-coordinates of a point (the one of the specified points that
#                   rescpects to the i-th column of pts) onto the camera's CCD w.r.t the coordinate frame of the camera. These
#                   coordinates are the coordinates of the perspective projection of the point on the CCD.
# depth_of_transformed_pts: a vector of length N (N is the size of the second dimension of the input array pts) whose i-th element
#                           is the depth (in 3D space w.r.t the camera's frame) of the respective 3D point corresponding to the
#                           i-th column of pts. 

def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    # Project the specified 3D points pts on the image plane, according to a pinhole
    # perspective projection model.
    N=pts.shape[1]

    # At first, transform the coordinates of the points given in the pts array, in order to get the coordinates of any point
    # w.r.t the camera's coordinate system. If the points are already given with their coordinates w.r.t the camera's coordinate
    # system, then by calculating the transformed_pts we just get the transpose of the pts array.
    transformed_pts=world2view(pts, R, t)

    # Compute the persective porjections, i.e the 2D-coordinates on the camera's CCD, of the points given in pts using a pinhole
    # perspective projection model. Also, save the depth of each 3D point in transformed_pts.
    pts_on_CCD=np.zeros((N,2))
    depth_of_transformed_pts=np.zeros(N)
    for i in range(N):
        if transformed_pts[i,2]!=0:
            pts_on_CCD[i,0]=(transformed_pts[i,0]*focal)/transformed_pts[i,2]
            pts_on_CCD[i,1]=(transformed_pts[i,1]*focal)/transformed_pts[i,2]
            depth_of_transformed_pts[i]=transformed_pts[i,2]

        else:
            # This means that the point has zeros distance from the center of the camera along the axis zc of the camera's
            # coordinate system. Thus, this point belongs to a vertical plane, declared by the unit vectors xc and yc of the
            # camera's coordinate system, which the center of the camera also belongs to. So, this point isn't caught by 
            # camera lens (it would be caught, only if the camera's CCD were infinity).
            pts_on_CCD[i,0]=np.inf
            pts_on_CCD[i,1]=np.inf
            depth_of_transformed_pts[i]=transformed_pts[i,2]

    return [pts_on_CCD, depth_of_transformed_pts]  
