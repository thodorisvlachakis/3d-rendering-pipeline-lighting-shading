import numpy as np
from TransformClass import *
# This function implements a world-to-view transform, which means that the function transforms a set of specified points
# (given as input), i.e transforms the coordinates, given with respect to some coordinate frame, of the specified points to
# the coordinates of the points w.r.t an another coordinate frame, that is defined by giving both the rotation matrix representing the
# rotation of the initial coordinate frame to the second coordinate frame and the shift vector that translates the reference point
# of the initial coordinate frame to the reference point of the second coordinate frame. Thus, the function can be used in order
# to transform the coordinates of a set od specified points w.r.t WCS to their coordintes w.r.t the coordinate frame of the camera.
# The camera frame is specified by the rotation with respect to the world frame (which is defined by the orthogonal axis x,y,z
# and the reference point is the O=[0 0 0]) and its point of reference, which is called c0 and means the center of the camera
# lens, whose coordinate are given with respect to the world frame.
# INPUTS:
# pts: a 3×N 2D-array whose each column includes the 3D-coordinates of a point (one of the initial specified points) with respect
#      to a coordinate frame.
# R: a 3×3 2D-array which represents the rotation matrix that stands for the rotation transform of the initial coordinate frame,
#    i.e the coordinate frame to which the coordinates of the specified points respect, to the second coordinate frame. When the
#    function is used to transform the coordinates of a set of specified points w.r.t WCS to their coordinates w.r.t the coordinate
#    frame of the camera, this array represents the rotation matrix of the camera w.r.t WCS.
# c0: a vector of length 3 that represents the coordinates of the reference point of the second coordinate frame, to which the 
#     coordinates of the (given) specified points will respect to, with respect to the initial coordinate frame. In other words,
#     this vector is the coordinates of the shift vector of the reference point of the initial coordinate frame to the reference
#     point of the new coordinate frame with respect to the initial coordinate frame. When the function is used to transform the
#     coordinates of a set of specified points w.r.t WCS to their coordinates w.r.t the coordinate frame of the camera, this vector
#     corresponds to the coordinates of the centr c0 of the camera w.r.t WCS.
#
# OUTPUTS:
# Transfrormed_pts: a N×3 2D-array whose i-th row includes the 3D-coordinates of a point (the one of the specified points that
#                   rescpects to the i-th column of pts) with respect to the new coordinate frame. In other words, this array
#                   includes the transformed coordinates of the specified points w.r.t the new specified coordinate frame.


def world2view(pts: np.ndarray, R: np.ndarray, c0 :np.ndarray)->np.ndarray:
    # Implements a world-to-view transform, i.e. transforms the specified
    # points to the coordinate frame of a camera. The camera coordinate frame
    # is specified rotation (w.r.t. the world frame) and its point of reference
    # (w.r.t. to the world frame).
    
    # In order to calculate the 3D-coordinates of a point w.r.t the camera's coordinate frame when its coordinates are given
    # w.r.t WCS, the expression below must be used:
    # c'=inv(R)*(c-c0) (1), where c are the 3D-coordinates of the point w.r.t the WCS, inv(R) is the inversion matrix of rotation
    # matrix R, which is equal to transpose(R) because R is rotation matrix, c0 are the 3D-coordinates of the reference point
    # of the camera's coordinate frame w.r.t WCS and c' are the 3D-coordinates of a point w.r.t the camera's coordinate frame.
    RT=np.transpose(R)
    v0=-np.dot(RT,c0)
    
    # The transformation of the points pts can easily be calculated if we use homogeneous coordinates. For extra simplicity
    # use an object of Transform class. We use the appropriate transformation matrix mat for the object by setting appropriately
    # the "rotation matrix" and the "translation vector" in order to represent the expression (1)
    transform=Transform()
    transform.mat[:3,:3]=RT
    transform.translate(v0)
    Transformed_pts=transform.transform_pts(np.transpose(pts))

    return Transformed_pts