import numpy as np
from lookat import *
from world2view import *
from perspective_project import *
from rasterize import *
from render_img import *
from calculate_normals import *
# This function is the implementation of photographing a 3D scene of an object using a camera. The function takes as inputs all
# the needed parameters that declares the position of the camera (i.e center of the camera, up vector, target point), the special
# characteristics of the camera (i.e focal length, the dimensions plane_w and plane_h of the camera plane) and the dimensions res_w
# and res_h of the image canvas which the object will display on, in order to set up the camera's coordinate system and be able
# to execute the necessary transoformations for any 3D point of the 3D object in 3D scene to be displayed to image canvas. The
# object in 3D scene is represented by 3D triangles that are defined by their 3D vertices. The color of the object is a consequence
# of the colors of the triangles it consists of. The color of each triangle is a consequence of the colors of the vertices that
# define the triangle. The function takes as inputs all these needed parameters, i.e a set of 3D coordinates of specific 3D points
# that define the triangles which define the 3D object, a set of indices that define the trianglea, i.e show the triads of the
# points above that define any triangle of the 3D object in 3D scene and a set of colors (RGB vectors) that respect to the points
# above. To implement the process of photographing a 3D scene of an object using a camera, the function computes the transformation
# of the given 3D points into points represented with their 2D image pixel coordinates (this process includes all the process of
# transforming the 3D coordinates of any point into camera's frame and then finding the projections on camera plane and rasterizing
# them to image pixel coordinates) and then the function uses the render_img method to generate the image using the above information 
# and calling either the "Gouraud" shading method or the "Phong" shading method (it depends on the value of the shader variable)
# giving also as input the set of colors that respect to the 3D points (as referred) and as a consequence respect to the
# corresponding 2D points on image canvas and of course the necessary information about illumination. Finally, note that the
# function takes as input the dimensions of the camera's CCD and the dimensions of the image canvas. The function colors only
# those triangles (of those given as input) that have all their vertices, when they have been projected on camera's plane,
# within the camera's plane (CCD) of dimensions H×W. This means that the function colors only the triangles that have all their
# vertices, after perspective projection and rasterization to pixel coordinates of image canvas, within the canvas of dimensions
# M×Ν. This work is implemented by render_img. 

# INPUTS:
# shader: A variable that specifies how to color the triangles, i.e. whether to use the shade_gouraud or shade_phong coloring
#         function. Valid values ​​it takes are “gouraud” (for shade_gouraud) and “phong” (for shade_phong).
# focal: a float number which represents the distance of the camera's CCD from the center of the camera (center of camera lens).
#        This is measured in units used by camera's coordinate system.
# eye: a vector of length 3 which represents the 3D-coordinates of the center of the camera w.r.t WCS.
# target: a vector of length 3 which represents the 3D-coordinates of the target point of the camera w.r.t WCS.
# up: a vector of length 3 which represents the 3D-coordinates of the up vector of the camera w.r.t WCS.
# bg_color: a vector of length 3 which represent the RGB vector (i.e the color) of the background which the canvas initializes to.
# M: an integer number that represents the width (i.e x-dimension that has the index 0 at the beginning of the range of width
#    when referring to indexing of positions by x-dimension) of image canvas.
# N: an integer number that represents the height (y-dimension that has the index 0 at the beginning of the range of height
#    when referring to indexing of positions by y-dimension) of image canvas.
# H: an integer number that represents the height (i.e y-dimension that has the index 0 at the middle of the range of height
#    when referring to indexing of positions by y-dimension) of camera plane.
# W: an integer number that represents the width (i.e x-dimension that has the index 0 at the middle of the range of width
#    when referring to indexing of positions by x-dimension) of camera plane.
# verts: a 3×Nu 2D-array whose its column includes the 3D-coordinates of a point (one of the initial specified points) with respect
#        to WCS.
# verts_colors: An array of dimension 3×Nu, each column of which includes the color coordinates (R,G,B) of the corresponding vertex of
#               the triangle declared in the corresponding column of the faces array.
# faces: A matrix of dimension 3×Nt, each column of which includes the three vertices that define a triangle. Each element of a
#        column of this array is a number, from 0 to Nu-1, that acts as a pointer to the corresponding column of the verts array
#        in which the coordinates of the vertices are stored.
# ka: a float number (it is between 0 and 1) which represents the coefficient of diffuse ambient light.
# kd: a float number (it is between 0 and 1) which represents the coefficient of diffuse reflection.
# ks: a float number (it is between 0 and 1) which represents the coefficient of specular reflection. 
# n: an integer that represents the Phong's constant used.
# l_pos: a Nl×3 2D-array whose each row includes the 3D-coordinates of a point light source.
# l_int: a Nl×3 2D-array whose i-th row includes intensity of the light radiation (it is a vector of length 3) that respects to
#        (i.e that is emmitted by) the point light source that corresponds to the i-th row of l_pos array.
# l_amb: a 3×1 vector which includes the components of the intensity of ambient light. Any of components belongs to interval 
#        [0,1].
#
# OUTPUTS:
# object_image: a 3-dimensional array N×M×3 which describes the outputed image. The third dimension of the array is a vector
#               of length 3 that represents the RGB color of the point (x,y) whose coordinates correspond to the first two
#               dimensions of the array (x corresponds to the second dimension and y to the first). This array contains all the
#               points of the final image defined by the Nt triangles that are given as input of the function, with the calculated
#               RGB vector for each point.


def render_object(shader, focal, eye, target, up, bg_color, M, N, H, W, verts, verts_colors, faces, ka, kd, ks, n, l_pos, l_int, l_amb)->np.ndarray:
    # render the specified object from the specified camera.

    # Declare the object image as a canvas which is initialized as totally white canvas. 
    object_image=np.ones((N, M, 3))
    # Define the color of the background on the canvas
    object_image[:]=bg_color

    # Firstly, set up the camera's coordinate system by using the given up vector, camera's target point and camera's eye
    # (i.e center). 
    [R_camera, t_camera]= lookat(eye, up, target)
    # The coordinates of t_camera are w.r.t the WCS and they are same with the coordinates of the center of the camera w.r.t
    # the WCS.

    # Secondly, transform the specified points, that define the object, in order to be represented in camrera's coordinate frame.
    transformed_v_pos= world2view(verts, R_camera, t_camera)

    # Third Step: Take a photo of the 3D scene where the object of interest is, using the camera setted up.
    # Compute the perspective projection of the points onto camera's plane. Also, save the depth of each point in 3D scene.
    # depth_of_trnsformed_v_pos: a vector of length Nu (Nu is the number of given points, i.e the size of the first dimension of
    #                            the input array verts) whose i-th element is the depth (in 3D space w.r.t the camera's frame)
    #                            of the respective 3D point corresponding to the i-th column of verts.
    # The variable depth_of_transformed_v_pos is very important, as it will be used while rendering of obeject image.
    [v_pos_perProj_2d, depth_of_transformed_v_pos]=perspective_project(verts, focal, R_camera, t_camera)
    
    # Fourth Step: Rasterize the computed 2D coordinates of the perspective projections of the points from the camera plane to
    # image pixel coordinates.
    scaled_v_pos_perProj_2d=rasterize(v_pos_perProj_2d, W, H, M, N)
    
    # Fifth Step: Compute the normal vectors of all the vertices included in verts as well as the 3D-coordinates of the center
    # of gravity for each triangle whose vertices are defined by the indices in columns of the array faces and their 3D-coordinates
    # correspond to the columns of the array verts.
    verts_normals=calculate_normals(verts, faces)
    # Transform the 3D-coordinates of any normal vector in verts_normals from WCS to camera's system.
    verts_normals_ccs=np.zeros((verts_normals.shape[0], verts_normals.shape[1])) # It is a 3×Nu array
    for i in range(verts_normals_ccs.shape[1]):
        verts_normals_ccs[:,i] = np.dot( np.transpose(R_camera) , (verts_normals[:,i] - t_camera) )

    triangles_b_coords=np.zeros((faces.shape[1], 3)) # It is a 3×Nt array
    triangles_b_coords_ccs=np.zeros((faces.shape[1], 3))
    for i in range(faces.shape[1]):
        f=faces[0,i]
        s=faces[1,i]
        t=faces[2,i]

        triangles_b_coords[i,:] = ( verts[:,f] + verts[:,s] + verts[:,t] ) / 3
        # Transform the 3D-coordinates of the center of gravity of the triangle from WCS to camera's system.
        triangles_b_coords_ccs[i,:] = np.dot( np.transpose(R_camera) , (triangles_b_coords[i,:] - t_camera) )
    
    # Final Step: Call render_img function in order to generate the object image by using the perspective projections of the
    # specified points on the camera plane, which have been rasterized to integer (digital) image pixel coordinates.
    # In order to call render_img, we need:
    # 1. A two dimensional F×3 array, whose each row includes indices referring to the columns of the array verts ,i.e to the points
    # whose 3D coordinates are saved in verts. Each row of this array defines the three vertices that declare a triangle in image
    # object. This triangle is defined in 3D space, but we have computed the perspective projections of all the points of verts
    # so we've got the "perspective projection" of the triangle onto camera plane and then the rasterized points to image pixel
    # coordinates. So, each row of this F×3 array includes indices referring to the rows of the array scaled_v_pos_perProj_2d,
    # i.e to the points of the 3D scene whose 2D coordinates (when persective projection on camera's plane and rastering to image
    # pixel coordinates is done) are saved to the corresponding (index points the number of the row) row of scaled_v_pos_perProj_2d.
    # Note: Indices are number for 0 to Nu-1 where Nu is the number of the points of interest. F represents the number of triangles
    # we use to render the object image, i.e F=Nt. -->This array is the np.transpose(faces).
    # 2. A two dimensional Nu×2 array, whose each row includes the 2D coordinates, w.r.t image pixel coordinates (i.e onto the image
    # canvas), of any point whose 3D coordinates are saved in rows of v_pos. -->This array is the scaled_v_pos_perProj_2d
    # 3. A vector of length Nu whose i-th element represents the depth of the point, that is saved by its 2D coordinates (w.r.t the
    # image pixel coordinates) in the i-th row of scaled_v_pos_perProj_2d, as it has been calculated by point's 3D coordinates
    # w.r.t the camera's coordinate system. --> This vector is the depth_of_transformed_v_pos.
    # 4. An array of dimension Nu×3, whose each row includes the color coordinates (R,G,B) of the corresponding vertex of
    # the triangle declared in the corresponding row of the faces array. -->This array is the np.transpose(vert_colors).
    # 5. A variable which will point either the phong or the gouraud method for coloring of image. --> This variable is the shader.
    # 6. An array of dimension Nu×3, whose i-th row includes the 3D coordinates, w.r.t camera's coordinate system, of the normal
    # vector of the vertex (of a triangle) whose 3D-coordinates are saved in i-th column of verts and so the rasterized 2D-coordinates
    # of it are stored in i-th row of scaled_v_pos_perProj_2d.--> This array is the np.transpose(verts_normals_ccs).
    # 7. An array of dimension Nt×3, whose i-th row includes the 3D coordinates, w.r.t camera's coordinate system, of the center
    # of gravity of the triangle which is defined by the vertices declared in the i-th column of the faces.
    # --> This array is the triangles_b_coords_ccs.
    # 8. A vector of length 3 which represents the 3D-coordinates (w.r.t WCS) of the observer of the object in 3D space, i.e
    # the 3D-coordinates of the camera w.r.t WCS. --> This vector is the eye.
    # 9. The needed information about illumination, i.e the needed information that is going to be input in the shade_gouraud or
    # shade_phong in render_img. --> ka, kd, ks, n, l_pos_ccs, l_int
    # For l_pos_css, transform the 3D-coordinates of any point light source in l_pos from WCS to camera's coordinate system.
    if len(l_pos.shape)==1:
        # This means that we have only one point light source.
        l_pos_ccs = np.dot( np.transpose(R_camera) , (l_pos - t_camera) )
    else:
        # This means that we have more that one point light sources or we have one point light source whose 3D-coordinates w.r.t
        # WCS are stored in an 2D array.
        l_pos_ccs=np.zeros((l_pos.shape[0] , l_pos.shape[1])) # It is a Nl×3 array
        for i in range(l_pos_ccs.shape[0]):
            l_pos_ccs[i,:] =  np.dot( np.transpose(R_camera) , (l_pos[i,:] - t_camera) )
    # 10. The dimensions M,N of the canvas in order to let the render_img know which triangles must shade and which mustn't.
    # 11. A vector of length 3 which represents the RGB color of the backgrounf on the canvas. --> This is the vector bg_color. 
    object_image=render_img(np.transpose(faces), scaled_v_pos_perProj_2d, np.transpose(verts_colors), depth_of_transformed_v_pos, 
                            shader, np.transpose(verts_normals_ccs), triangles_b_coords_ccs, eye, ka, kd, ks, n, l_pos_ccs, l_int, l_amb, M, N, bg_color)

    return object_image