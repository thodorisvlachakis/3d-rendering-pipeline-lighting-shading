import numpy as np
# This function rasterizes the 2D coordinates of a given set of points from the camera plane to image pixel coordinate system.
# In fact, any point on the camera plane is described by its 2D coordinates w.r.t the camera plane. Camera's plane (i.e CCD of
# camera) can be modeled as a plane with dimensions plane_w × plane_h and the property that the axis of the camera passes through
# the center of this plane (i.e the center of the camera respects to the center of the plane) and within this range of horizontal
# and vertical coordinates, respectively, the 2D coordinates (on the camera plane) of each 3D point, that is "caught" by camera
# lens, belong. So, this function makes a proper mapping of these 2D coordinates of any point on the camera plane to 2D
# coordinates w.r.t image pixel coordinates. Note that image is a digital canvas, i.e a canvas divided into horizontal and
# vertical positions that respect to integer 2D coordinates and the indexing of them starts from down to up and from right to left
# and has values [0,...,res_w] and [0,..., res_h] respectively, with dimensions res_w × res_h. Thus, the function executes a 
# proper shift and re-scaling of the given 2D coordinates of any point on camera plane to 2D coordinates of the point on image
# canvas. Note that this execution may result in non-integer coordinates. Thus, a round is needed.

# INPUTS:
# pts_2d: a N×2 2D-array whose its row includes the 2D-coordinates of a point displayed on camera plane.
# plane_w: an integer number that represents the width (i.e x-dimension that has the index 0 at the middle of the range of width
#          when referring to indexing of positions by x-dimension) of camera plane.
# plane_h: an integer number that represents the height (i.e y-dimension that has the index 0 at the middle of the range of height
#          when referring to indexing of positions by y-dimension) of camera plane.
# res_w: an integer number that represents the width (i.e x-dimension that has the index 0 at the beginning of the range of width
#        when referring to indexing of positions by x-dimension) of image canvas.
# res_h: an integer number that represents the height (y-dimension that has the index 0 at the beginning of the range of height
#        when referring to indexing of positions by y-dimension) of image canvas. 
#
# OUTPUTS:
# Transfrormed_pts: a N×2 2D-array whose i-th row includes the 2D-coordinates of a point (the point represented by its 2D coordinates,
#                   w.r.t camera plane, at the i-th row of pts_2d) image canvas, i.e the 2D (digital) pixel coordinates on image
#                   of this point.


def rasterize(pts_2d: np.ndarray, plane_w: int, plane_h: int, res_w: int, res_h: int)->np.ndarray:
    # Rasterize the incoming 2d points from the camera plane to image pixel coordinates
    # The camera plane has dimensions plane_h x plane_w, but the image canvas has dimensions res_h x res_w.
    # Re-scale the given points from camera plane to image canvas. Firstly, shift proprerly the points in order to
    # get the center of the camera plane, ehich is the point with 2D-coordinates [0 0] w.r.t camera's frame, at the
    # center of the image canvas, which is not the [0 0] of the canvas. Similarly, do for all the points.
    N=pts_2d.shape[0]

    scale_x=res_w / plane_w
    scale_y=res_h / plane_h
    shifted_pts_2d=pts_2d + np.array([plane_w / 2,plane_h / 2])
    
    scaled_pts_2d=np.zeros((N,2))
    scaled_pts_2d[:,0]=shifted_pts_2d[:,0] * scale_x
    scaled_pts_2d[:,1]=shifted_pts_2d[:,1] * scale_y
    # The image canvas has integer coordinates. Thus, the coordinates of scaled points must be integers sο that they
    # can correspond to integer positions (pixels) of the image canvas. Hence, round the coordinates saved in scaled_pts_2d.
    for i in  range(N):
        scaled_pts_2d[i,:]=[ round(scaled_pts_2d[i,0]), round(scaled_pts_2d[i,1]) ]
    scaled_pts_2d=scaled_pts_2d.astype('int32')

    return scaled_pts_2d
