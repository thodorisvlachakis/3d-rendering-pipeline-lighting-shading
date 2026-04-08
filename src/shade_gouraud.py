import numpy as np
import math
from compute_lines_triangle import *
from sort_vertices import *
from vector_interp import *
from light import *
# This function takes as input an image that is a canvas of dimension M×N and an RGB vector corresponds to every point of this
# canvas. Also, the function takes as input the integer coordinates of the three vertices of a triangle, after its perspective
# projection on camera's plane and the RGB colors of its three vertices. Moreover, the normal vectors that correspond to these
# three vertices of the triangle, before its perspective projection to the camera's plane, are given. Furthermore, the function
# takes, as input, information about the illumination of this triangle. The positions in 3D space, where the point light sources
# are located, are given as well as the intensity of the light radiation emitted by each of them is given. The coefficients of
# diffuse ambient light, diffuse reflection and specular reflection for the vertices of the triangle are considered as known and
# given as inputs.
# Note that the coordinates of the three vertices of the triangle are 2D-coordinates and they refer to the coordinates of the
# vertices after perspective projection on camera's plane. Also, the 3D-coordinates of the cevnter of gravity of the triangle,
# before the perspective projection of the triangle on camera's plane, are given. Using this information, the function calculates
# the 3D-coordinates of the three vertices of the triangle, before the perspective projection on camera's plane, solvinge a 3×3
# linear system of equations. The 3D-coordinates of the vertices before the perspective projection of the triangle on camera's
# plane are needed in order to use the function light to calculate the intensity of trichromatic radiation reflected by these
# three vertices of the trianlge.
# Having this triangle, i.e the 2D-coordinates of its vertices and information to calculate the 3D-coordinates of them, the
# three RGB vectors that respect to them, the normal vectors (before perspective projection) that respect to them and the needed
# information for the illumination of the triangle, function calculates the intensity of the trichromatic light radiation, which
# is a vector I, reflected by the three vertices of the trianlge. Having these three vectors of intensity of trichromatic radiation
# for each vertx of the triangle, the function also calculates the intensity of the trichromatic light radiation that corresponds 
# to every point that belongs to the trianlge. This trichromatic radiation vector is computed for each point, by using vector_interp
# function giving as inputs the active points that belong to the same searching line y with the point and their trichromatic radiation
# vectors I. The trichromatic radation vector I of an active point is computed by using vector_interp function giving as inputs
# the vertices of the triangle that determine the active line which the active point belong to and their trichromatic radiation
# vectors. Function shades the triangle giving as RGB vector to its points the vector that represents the intensity of trichromatic
# radiation reflected by these points, following the above logic called "Gouraud Shading".
# So, the image that is outputed by function is same with inputed image apart from this (given) triangle that is shaded with
# the color represented by the calculated RGB vector. 

# INPUTS:
# verts_p: a 3×2 2D-array whose each row contains the 2-dimensinal coordinates of a vertex of a 3D triangle, after its perspective
#          projection on camera's plane
# verts_n: a 3×3 2D-array whose i-th column includes the 3D-coordinates of the normal vector of the surface, i.e the triangle,
#          calculated at the point/vertex of the trainlge whose 2D-coordinates, after its perspective projection on camera's plane,
#          correspond to the i-th row of the array verts_p. They are considered w.r.t camera coordinate system.
# verts_c: a 3×3 2D-array whose each row contains the RGB color (values belong to interval [0,1]) of the corresponding vertex
#          of the triangle.
# bcoords: a 3×1 column vector which includes the 3D-coordinates of the point that represents the center of gravity of the 3D
#          triangle, before its perspective projection on camera's plane. They are considered w.r.t camera coordinate system.
# cam_pos: a 3×1 column vector which includes the 3D-coordinates wr.t WCS of the point where the observer (i.e the camera) is
#          located.
# ka: a float number (it is between 0 and 1) which represents the coefficient of diffuse ambient light.
# kd: a float number (it is between 0 and 1) which represents the coefficient of diffuse reflection.
# ks: a float number (it is between 0 and 1) which represents the coefficient of specular reflection. 
# n: an integer that represents the Phong's constant used.
# f: a float number that represents the focal length of the camera.
# l_pos: a Nl×3 2D-array whose each row includes the 3D-coordinates of a point light source. They are considered w.r.t camera
#        coordinate system.
# l_int: a Nl×3 2D-array whose i-th row includes intensity of the light radiation (it is a vector of length 3) that respects to
#        (i.e that is emmitted by) the point light source that corresponds to the i-th row of l_pos array.
# l_amb: a 3×1 vector which includes the components of the intensity of ambient light. Any of components belongs to interval 
#        [0,1].
# img : a 3-dimensional array N×M×3 which describes the inputed image that is wanted to be updated by the function "shade_gouraud".
#       The third dimension of the array is a vector of length 3 that represents the RGB color of the point (x,y) whose
#       coordinates correspond to the first two dimensions of the array (x at the second and y at the first column, beacause M
#       represents the heigth and N represents the width of the canvas). Any pre-existing triangles may be presentin img
#
# OUTPUTS:
# updated_img: a 3-dimensional array N×M×3 which describes the outputed image. The third dimension of the array is a vector
#              of length 3 that represents the RGB color of the point (x,y) whose coordinates correspond to the first two
#              dimensions of the array (x at the second and y at the first column). This array contains all the points of the
#              triangle with the calculated RGB vectors and the pre-existing points of the input image, i.e the array img. The
#              pre-existing RGB vectors for points that corresponds to the points of the triangle are getting overlapped by
#              calculated RBG vectors.

def shade_gouraud(verts_p, verts_n, verts_c, bcoords, cam_pos, ka, kd, ks, n, f, l_pos, l_int, l_amb, img)->np.ndarray:
    # verts_p includes the 2D-coordinates of the vertices of the triangle (after perspective projection on camera's plane).
    
    updated_img=img
    M=img.shape[1]
    N=img.shape[0]
    
    sorted_vertices_by_y=sort_vertices(verts_p,2)
    
    # vcolors_sorted_vertices: an array of dimension 3×3 whose each row contains the RGB color (values belong to interval [0,1]) of the
    #                          corresponding vertex of the triangle, when vertices are sorted.
    # normals_sorted_vertices: an array of dimension 3×3 whose each column contains the 3D-coordinates of the normal vector that
    #                          corresponds to each vertex of the triangle, when vertices are sorted.
    # Compute vcolors_sorted_vertices and normals_sorted_vertices.
    vcolors_sorted_vertices=np.ones((3,3))
    normals_sorted_vertices=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if (sorted_vertices_by_y[i,:]==verts_p[j,:]).all():
                vcolors_sorted_vertices[i,:]=verts_c[j,:]
                normals_sorted_vertices[:,i]=verts_n[:,j]
    
    # Now, compute the array I_trichromatic_sorted_vertices, using the overall light model represented by function light.
    # I_trichromatic_sorted_vertices: an array of dimension 3×3 whose each column contains the intensity of the trichromatic light
    #                                 radiation reflected by each vertex of the triangle, when vertices are sorted.
    # To compute the intnsity of the trichromatic light radiation reflected by a point of interest, the 3D-coordinates of this
    # point are needed, in order to use the function light.
    I_trichromatic_sorted_vertices=np.zeros((3,3))
    sorted_vertices_by_y_3D=np.zeros((3,3))
    
    # **It's all about maths... **
    A=np.ones((3,3))
    for i in range(3):
        A[0:2,i]= sorted_vertices_by_y[i,:] / f    
    
    det_A=np.linalg.det(A)
    if det_A==0:
        # THen A is signular, so we do an estimation using Least Squares approach.
        vector1, _, _, _ = np.linalg.lstsq(A, (3*bcoords), rcond=None)
    else:
        vector1 = 3*( np.dot(np.linalg.inv(A) , bcoords) )
    
    for i in range(3):
        # At first, compute the 3D-coordinates of the vertices of the triangle (before perspective projection) using the 2D-coordinates
        # and the 3D-coordinates of the center of gravity of the triangle. Using this information, the vector1 has been calculated
        # and it refers to z-dimensions of the vertices of the triangle, when vertices are sorted. We can use vector1 and the 2D-coordinates
        # of the vertices after perspective projection on camera's plane in order to find their 3D-coordinates, before perspective
        # projection. We also need the focal length of the camera.
        sorted_vertices_by_y_3D[i,0]=sorted_vertices_by_y[i,0]*vector1[i]/f
        sorted_vertices_by_y_3D[i,1]=sorted_vertices_by_y[i,1]*vector1[i]/f
        sorted_vertices_by_y_3D[i,2]=vector1[i]

        I_trichromatic_sorted_vertices[:,i]=light(sorted_vertices_by_y_3D[i,:], normals_sorted_vertices[:,i], vcolors_sorted_vertices[i,:], np.zeros(3), ka, kd, ks, n, l_pos, l_int, l_amb)

    # Compute the lines that pass through the vertices of the triangle
    # Because of sorting the vertices both the first and the last row of triangle_lines refer to the vertex with the minimum y
    # The first and the second row of triangle_lines refer to the vertex with the "medium" y
    # The second and the last row of triangle_lines refer to the vertex with the maximum y
    triangle_lines=compute_lines_triangle(sorted_vertices_by_y)

    # Compute ymin and ymax for searching lines
    ymin=sorted_vertices_by_y[0,1]
    ymax=sorted_vertices_by_y[2,1]

    # active_lines is a vector of length 3. If its i element is 1 then the i-th line is active for the searching line y.
    # active_lines will be updated during the algorithm execution.
    # active_points is an array of dimension 2×3 whose each row contains the coordinates of the point of intersection of the
    # searching line with the line inclued in some row of triangle_lines, at the first two columns. At the third column there
    # is the characteristic number for the active line which the active point belong to. The characteristic number is: 
    # 0 for active line between vertices "1" and "2" (sorted), 1 for active line between vertices "2" and "3" (sorted),
    # 2 for active line between vertices "1" and "3" (sorted).
    # active_points will be updated during the algorithm execution.
    #active_points=np.array((2,3))

    # Compute active lines for the searching line y=ymin
    active_lines=np.array([1,0,1])

    # Since we have triangle the number of active points for every searching line y between ymin and ymax is 2. For y=ymin
    # and y=ymax there is only one active point (since y min and ymax are integers). For these cases we assume that the
    # active point is double.
    # Compute active points for the searching line y=ymin

    active_points=np.array([ [sorted_vertices_by_y[0,0],sorted_vertices_by_y[0,1], 0],
                           [sorted_vertices_by_y[0,0],sorted_vertices_by_y[0,1], 2] ])

    # Check if there are horizontal lines and save this information
    horizontal_lines=np.array([0,0,0])
    for i in range(3):
        if triangle_lines[i,0]==0:
            horizontal_lines[i]=1
            # If a line is horizontal, exclude it from active_lines
            active_lines[i]=0

    # Special case: All the vertices are in the same horizontal line, so there is no actual triangle
    # This is a special case as well as the case where the vertices are in the same vertical line but this case is contained below
    # instead of this case where (1/m)=infinity for all the triangle lines.
    if (horizontal_lines[0]==1 and horizontal_lines[1]==1) or (horizontal_lines[0]==1 and horizontal_lines[2]==1) or (horizontal_lines[1]==1 and horizontal_lines[2]==1):
        sorted_vertices_by_x=sort_vertices(sorted_vertices_by_y,1)
        x1=sorted_vertices_by_x[0,0]
        x2=sorted_vertices_by_x[2,0]
        xm=sorted_vertices_by_x[1,0]
        sorted_vertices_by_x_3D=np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                if (sorted_vertices_by_x[i,:]==verts_p[j,:]).all():
                    vcolors_sorted_vertices[i,:]=verts_c[j,:]
                    normals_sorted_vertices[:,i]=verts_n[:,j]
            
            sorted_vertices_by_x_3D[i,0]=sorted_vertices_by_x[i,0]*vector1[i]*f
            sorted_vertices_by_x_3D[i,1]=sorted_vertices_by_x[i,1]*vector1[i]*f
            sorted_vertices_by_x_3D[i,2]=vector1[i]
            
            I_trichromatic_sorted_vertices[:,i]=light(sorted_vertices_by_x_3D[i,:], normals_sorted_vertices[:,i], vcolors_sorted_vertices[i,:], np.zeros(3), ka, kd, ks, n, l_pos, l_int, l_amb)
        for x in range(x1,x2+1):
            if x<=xm:
                color_RGB=vector_interp(sorted_vertices_by_x[0,:],sorted_vertices_by_x[1,:],I_trichromatic_sorted_vertices[:,0],I_trichromatic_sorted_vertices[:,1],x,1)
            else:
                color_RGB=vector_interp(sorted_vertices_by_x[1,:],sorted_vertices_by_x[2,:],I_trichromatic_sorted_vertices[:,1],I_trichromatic_sorted_vertices[:,2],x,1)
            updated_img[N-1-sorted_vertices_by_y[0,1],x,:]=color_RGB
        
        return updated_img
    
    # If the line between vertices "1" and "2" (sorted) is horizontal, start scanning from ymin +1 and update active_lines and
    # active_points
    if horizontal_lines[0]==1:
        ymin=ymin+1
        active_lines[1]=1
        # First active point belongs to the line between vertices "1" and "3" (sorted)
        if(np.isinf(triangle_lines[2,0])==False):
            # This is a non-vertical active line
            # Update the active point
            active_points[0,1]=ymin
            active_points[0,0]=active_points[0,0] + 1/triangle_lines[2,0]
            active_points[0,2]=2
        else:
            # This is a vertical active line
            active_points[0,1]=ymin
            active_points[0,0]=active_points[0,0]
            active_points[0,2]=2
        
        # Second active point belongs to the line between vertices "2" and "3" (sorted). This is active point will be calculated
        # by using the vertex "2" (as initial point of this line) which belongs to the same horizontal line with vertex "1" (sorted).
        if(np.isinf(triangle_lines[1,0])==False):
            # This is a non-vertical active line
            # Update the active point
            active_points[1,1]=ymin
            active_points[1,0]=sorted_vertices_by_y[1,0] + 1/triangle_lines[1,0]
            active_points[1,2]=1
        else:
            # This is a vertical active line
            active_points[1,1]=ymin
            active_points[1,0]=sorted_vertices_by_y[1,0]
            active_points[1,2]=1
    

    # active_colors: an array of dimension 2×3 whose i-th row contains the RGB color (vector) of the i-th active point from the
    #                array active_points. This RGB color is calculated via vector_interp function giving as inputs the vertices
    #                of the triangle that determine the active line which these active points belong to and the vectors that are
    #                corresponding to these vertices and represent the intensity of the trichromatic light radiation reflected by
    #                each of them. 
    active_colors=np.ones((2,3))

    # Start scanning by searching lines

    for y in range(ymin,ymax+1):
        # sort active_points by x
        active_points=sort_vertices(active_points,1)
        cross_count=0
        update_now=0

        # Declare the active_colors for the searching line y
        for i in range(2):
            if active_points[i,2]==0:
                # The active point belongs to the line between vertices "1" and "2" (sorted)
                if sorted_vertices_by_y[0,1]==sorted_vertices_by_y[1,1]:
                    active_color1=vector_interp(sorted_vertices_by_y[0,:],sorted_vertices_by_y[1,:], I_trichromatic_sorted_vertices[:,0],
                                                I_trichromatic_sorted_vertices[:,1],active_points[i,0],1)
                else:
                    active_color1=vector_interp(sorted_vertices_by_y[0,:],sorted_vertices_by_y[1,:], I_trichromatic_sorted_vertices[:,0],
                                                I_trichromatic_sorted_vertices[:,1],active_points[i,1],2)
        
            elif active_points[i,2]==1:
                # The active point belongs to the line between vertices "2" and "3" (sorted)
                if sorted_vertices_by_y[1,1]==sorted_vertices_by_y[2,1]:
                    active_color1=vector_interp(sorted_vertices_by_y[1,:],sorted_vertices_by_y[2,:], I_trichromatic_sorted_vertices[:,1],
                                                I_trichromatic_sorted_vertices[:,2],active_points[i,0],1)
                else:
                    active_color1=vector_interp(sorted_vertices_by_y[1,:],sorted_vertices_by_y[2,:], I_trichromatic_sorted_vertices[:,1],
                                                I_trichromatic_sorted_vertices[:,2],active_points[i,1],2)

            elif active_points[i,2]==2:
                # The active point belongs to the line between vertices "1" and "3" (sorted)
                if sorted_vertices_by_y[0,1]==sorted_vertices_by_y[2,1]:
                    active_color1=vector_interp(sorted_vertices_by_y[0,:],sorted_vertices_by_y[2,:], I_trichromatic_sorted_vertices[:,0],
                                                I_trichromatic_sorted_vertices[:,2],active_points[i,0],1)
                else:
                    active_color1=vector_interp(sorted_vertices_by_y[0,:],sorted_vertices_by_y[2,:], I_trichromatic_sorted_vertices[:,0],
                                                I_trichromatic_sorted_vertices[:,2],active_points[i,1],2)
            active_colors[i,:]=active_color1


        # Start scanning of search line y
        for x in range (M):
            cross_count=0
            for i in range(active_points.shape[0]):
                if x>=active_points[i,0]:
                    cross_count=cross_count+1
            if (cross_count % 2) !=0:
                # Draw pixel
                # Use the "Gouraud Shading" logic which determines the color by the linear interpolation of the colors of the
                # active points that belong to this searching line. Since cross_count is odd (Draw pixel) the active points of
                # this searching line haven't the same x coordinate. So, we call vector_interp giving x-dimension of the point.
                color_RGB=vector_interp(active_points[0,0:2],active_points[1,0:2],active_colors[0,:],active_colors[1,:],x,1)
                updated_img[N-1-y,x,:]=color_RGB
        
        # Update the list of active_lines for the next searching line
        
        # First case: The line between vertices "1" and "2" (sorted) is active, but for the next searching line must be excluded.
        # 1st Sub-case: The line between vertices "2" and "3" (sorted) must be included, if and only if is not horizontal. 
        if y<sorted_vertices_by_y[1,1] and (y+1)>=sorted_vertices_by_y[1,1] and horizontal_lines[1]!=1:
            active_lines[0]=0
            active_lines[1]=1
            # Declare a variable to know when update the list of active_lines and which case it is.
            update_now=1
        # 2nd Sub-case: The line between vertices "2" and "3" (sorted) is horizontal, which means that this line is same with 
        # the searcing line y+1 and we assume that it belongs to the triangle "above" the triangle we shade.
        if y<sorted_vertices_by_y[1,1] and (y+1)>=sorted_vertices_by_y[1,1] and horizontal_lines[1]==1:
            active_lines[0]=0
            active_lines[1]=0
            active_lines[2]=0
            # Declare a variable to know when update the list of active_lines and which case it is.
            update_now=2

        # Second case: What happens when y+1=ymax? 
        # 1st Sub-case: Searching line y has active lines both the line between vertices "2" and "3" and the line between vertices
        # "1" and "3" (sorted), respectively. This means that the line between vertices "2" and "3" (sorted) is not horizontal.
        if y+1==ymax and horizontal_lines[1]!=1:
            active_points=np.array([ [sorted_vertices_by_y[2,0],sorted_vertices_by_y[2,1], 1],
                           [sorted_vertices_by_y[2,0],sorted_vertices_by_y[2,1], 2] ])
            update_now=3

        # 2nd Sub-case: Searching line y has active lines both the line between vertices "1" and "2" and the line between vertices
        # "1" and "3" (sorted), respectively. This means that the line between vertices "2" and "3" (sorted) is horizontal. This
        # case is the 2nd Sub-Case of the First case already implemented.

        # Last Case: The current searcing line is y=ymax and there is no need to update for the next searching line
        if y==ymax:
            update_now==4

        # Update the list of active_points for the next searching line
        if update_now==0:
            # This means no update occurred.
            # Find the active line which each active point belongs to and update the active_points for the next searching line.
            for i in range(2):
                for j in range(len(active_lines)):
                    if active_lines[j]==1:
                        if active_points[i,2]==j:
                            # The active line, which the active point belongs to, is found
                            if (np.isinf(triangle_lines[j,0])==False):
                                # This means that the active line is non-vertical
                                active_points[i,0]=active_points[i,0] + 1/triangle_lines[j,0]
                                active_points[i,1]=y+1
                            else:
                                # This means that the active line is vertical
                                # Update only the y-coordinate.
                                active_points[i,1]=y+1

        elif update_now==1:
            # This means that the line between vertices "2" and "3" (sorted) is now active for y+1
            # Find the active point that belongs to the line between vertices "1" and "2" (sorted), because this has to be excluded.
            # Use the characteristic number of the point !!!
            for i in range(2):
                if(active_points[i,2]==0):
                    # The active point, which belongs to the line between vertices "1" and "2" (sorted) and must be excluded, is found.
                    # Include the vertex "2" (sorted) in the active_points.
                    active_points[i,0]=sorted_vertices_by_y[1,0]
                    active_points[i,1]=sorted_vertices_by_y[1,1]
                    active_points[i,2]=1
                elif(active_points[i,2]==2):
                    # This active point belongs to the line between vertices "1" and "3" (sorted) and will be updated classically.
                    if (np.isinf(triangle_lines[2,0])==False):
                         # This means that the active line is non-vertical
                        active_points[i,0]=active_points[i,0] + 1/triangle_lines[2,0]
                        active_points[i,1]=y+1
                    else:
                        # This means that the active line is vertical
                        # Update only the y-coordinate.
                        active_points[i,1]=y+1
                        
        elif update_now==2:
            # In this case the line between vertices "2" and "3" is horizontal and it is assumed that belongs to the "above"
            # triangle, so the scanning process is over.
            y=ymax # This will terminate the "for" loop.
    
    # Last part of shading: Decide on horizontal (if it exists) line between two of three vertices of the triangle.
    # Beacause of "agreement", we assume that if a horizontal line determines an "upper border" of the triangle is not
    # considered to belong to the triangle (it belongs to the "above" triangle). So, if the line between vertices "1" and "2"
    # (sorted) is horizontal , it is considered as part of the triangle. If the line between vertices "2" and "3" (sorted) is 
    # horizontal, it is ont considered as part of the triangle.
    # Note: Because of sorting the vertices by y, the line between vertices "1" and "3" (sorted) can't be horizontal !!!
    if horizontal_lines[0]==1:
        # This means that the vertices "1" and "2" (sorted) has the same y(=ymin) coordinate.        
        if sorted_vertices_by_y[0,0]<=sorted_vertices_by_y[1,0]:
            x1=sorted_vertices_by_y[0,0]
            x2=sorted_vertices_by_y[1,0]
        else:
            x1=sorted_vertices_by_y[1,0]
            x2=sorted_vertices_by_y[0,0]
        for x in range(x1,x2):
            color_RGB=vector_interp(sorted_vertices_by_y[0,:],sorted_vertices_by_y[1,:],I_trichromatic_sorted_vertices[:,0],
                                    I_trichromatic_sorted_vertices[:,1],x,1)
            updated_img[N-1-sorted_vertices_by_y[0,1],x,:]=color_RGB

    return updated_img