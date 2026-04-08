import numpy as np
# This function computes the three lines that pass through the vertices (two each time) of a given triangle
# INPUTS:
# vertices : an integer array of dimension 3×2 whose each row contains the 2-dimensinal coordinates of a vertex of the triangle.
# OUTPUTS:
# triangle_lines: an array of dimension 3×2 whose each row refers to a line that pass through two of the three vertices of the
#                 triangle. At the first column has the value of the slope m (if inf, this means that m is infinite) and at
#                 the second column has the value of the constant term b (if inf, this means that m is infinite). 

def compute_lines_triangle(vertices):
    triangle_lines=np.zeros((3,2),dtype=float)
    for i in range(3):
        if i==2:
            # compute the line that passes through the vertices: vertices[0,:] and vertices[2,:]
            Dy=vertices[2,1]-vertices[0,1]
            Dx=vertices[2,0]-vertices[0,0]
            if Dx==0:
                m=np.inf
                b=np.inf
            else:
              m=Dy/Dx
              b=vertices[0,1]-m*vertices[0,0]

        else:
            # compute the line that passes through the vertices: vertices[i,:] and vertices[i+1,:]
            Dy=vertices[i+1,1]-vertices[i,1]
            Dx=vertices[i+1,0]-vertices[i,0]
            if Dx==0:
                m=np.inf
                b=np.inf
            else:
              m=Dy/Dx
              b=vertices[i,1]-m*vertices[i,0]

        triangle_lines[i,0]=m
        triangle_lines[i,1]=b  

    return triangle_lines  