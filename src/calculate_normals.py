import numpy as np

# This function calculates the normal vectors corresponding to the vertices of a surface, which defines a 3D object. It is
# assumed that a three-dimensional object consists exclusively of Nt three-dimensional triangles or in other words the object is
# defined by the surface which is composed of these triangles. Also, it is assumed that each triangle is fully defined by its
# three vertices. For these three points/vertices of the triangle, the normal vectors of the surface, calculated at these vertices,
# can be defined. Considering these assumptions, the function calculates the normal vector corresponding to each vertex of the
# surface using the property that this vertex is one of the three that define a tringle, which is a part of definition of the
# surface. It is probable that a vertex belongs to more than one different triangles (when it is common vertex). In this case,
# the function calculates for this specific point as many normal vectors as the number of the triangles to which this vertex 
# belongs. If such a common vertex exists, its 3D coordinates will be included in the 3×Nu 2D-array, that is given as input of
# the function as the array with the coordinates of the vertices of the surface and called "verts", as many times (i.e in as
# many columns of this array) as the number of the triangles to which this vertex belongs, so that more than one normal vectors
# at this vertex can be calculated. Each of them refers to an another triangle of the surface that defines the 3D object. 
# The function uses the following property in order to define the normal vector at each vertex of a single triangle of the
# surface. Any of the triangles, that define the surface, is declared by its three 3D vertices. The function takes as input a
# 3×Nt 2D-array, called "faces", whose each column includes indexes (i.e integers between 1 and Nu), each of them refers to a
# column of the array "verts". Each column of "faces" defines a triangle of the surface. It is assumed that every trianlge is 
# also a surface which is defined with some outer and some inner side. This side is declared at any point of the triangle by
# the direction of the normal vector that is calculated at this point. The function assumes that each triangle, declared in
# "faces", is given in such a way that the order in which the vertices are listed in the column indicates with the right-hand
# screw rule the direction of the normal vector and thus towards what is the outer side of the surface, i.e the object. So, if
# we 've got, as given input of the function, that the 3D points p1, p2, p3 deifne a triangle and they are given with this order
# in "faces", the function can calculate the normal vectors that respect to each of them for this triangle using the property
# that the vectors p1p2 (defined by p1 and p2) and p1p3 (defined by p1 and p3) define the plane to which the triangle belongs
# to. Since the order in which the vertices p1, p2, p3 are listed in a column of "faces" is this, then the normal vector n1 of
# the traingle calculated at the point p1 is given by the cross product of p1p2 and p1p3, according to the right-hand screw
# rule. In the same way, the normal vector n2 at he point p2 is given by the cross product of p2p3 and p2p1 and the normal
# vector n3 at the point p3 is given by the cross product of p3p1 and p3p2. However, since the trianlge belongs to a plane, all
# its vertices belong to the same plane, so only one of the above cross products is needed, because the normal vector is the
# same at each point of the triangle as well as at its vertices. The normal vectors are unit vectrors, so the function 
# standarize each of them. The function do similar calculation for each triangle included in "faces", resulting in the normal
# vectors of each point/vertex in "verts".


# INPUTS:
# verts: a 3×Nu 2D-array whose each column includes the 3D-coordinates of a point/vertex of the surface that define the 3D oblect
# faces: an 3×Nt 2D-array whose k-th column includes the ordinal numbers of the vertices of the k-th triangle of the object.
#        Each element (ordinal number) of this column is a number, from 0 to Nu-1, that acts as a pointer to the corresponding
#        column of the verts array in which the coordinates of the vertices are stored.
#
# OUTPUTS:
# normals: a 3×Nu 2D-array whose i-th column includes the 3D-coordinates of the normal vector of the surface that respects to
#          the point (i.e the vertex (one of them that define the surface) ) of the surface that is corresponding to the i-th
#          column of the verts (i.e its 3D-coordinates are given in the i-th column of verts). 

def calculate_normals(verts, faces)->np.ndarray:
    Nu=verts.shape[1]
    Nt=faces.shape[1]

    normals=np.zeros((3,Nu))

    # Define an algorithm that calculates the normal vectors at the three vertices which define a triangle. Do the same for all
    # the triangles delcared in faces.
    # Define a 3×3 2D array vertices whose i-th column includes the 3D-coordinates that refer to the vertex corresponding to
    # the i-th row of faces in each k iteration of the algorithm (where i is between 0 and 2 and k is between 0 and Nt-1).
    # Update the array vertices in every iteration, in order to work in an another triangle.
    vertices=np.zeros((3,3))
    for i in range(0,Nt):
        indexes=faces[:,i]
        for j in range (3):
            vertices[:,j]=verts[:,indexes[j]]
            
        # In order to calulate normal vectors, define the vectors between the vertices of the triangle, which are the vectros
        # that define the plane to which the triangle belongs.
        vector1=vertices[:,1]-vertices[:,0]
        vector2=vertices[:,2]-vertices[:,0]
        # These vectors define the plane in which the triangle belong. So, their cross product define the normal vector in every
        for k in range(3):
            # Define normal vectors as unit vectors
            normals[:,indexes[k]]= np.cross(vector1,vector2) / np.linalg.norm(np.cross(vector1,vector2))

    return normals