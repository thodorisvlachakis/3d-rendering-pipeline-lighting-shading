# This function sorts a set of given vertices by x or y depending on specific input
# INPUTS:
# vertices : an integer array of dimension N×2 whose each row contains the 2-dimensinal coordinates of a vertex.
#            Also, this integer array can be an integer array of dimension N×3 whose each row contains the 2-dimensional
#            coordinates of a vertex at the first two columns and the characteristice number of the vertex at the third column.
# index: an integer whose value decides if the vertices will be sorted by x (if index=1) or y (if index=2). Valid values are 1 and 2.
# OUTPUTS: 
# sorted_vertices : an integer array of dimension N×2 whose each row contains the 2-dimensinal coordinates of a vertex. This
#                   array has the rows of array "vertices" sorted by x or y (depending on index)

def sort_vertices(vertices,index):
    if index!=1 and index!=2:
        return "Invalid value of variable called index."
    sorted_vertices=vertices
    n=vertices.shape[0]
    if index==1:
        # Sort the vertices by x
        for j in range(n-1):
            s=-1
            min=sorted_vertices[j,0]
            for i in range(j+1,n):
                if sorted_vertices[i,0]<min:
                    min=sorted_vertices[i,0]
                    # declare minimum's position
                    s=i
            # swap
            if s!=-1:
                # This means that a swap is needed
                sorted_vertices[[j,s]]=sorted_vertices[[s,j]]  
        
    elif index==2:
        # Sort the vertices by y
        for j in range(n-1):
            s=-1
            min=sorted_vertices[j,1]
            for i in range(j+1,n):
                if sorted_vertices[i,1]<min:
                    min=sorted_vertices[i,1]
                    # declare minimum's position
                    s=i
            # swap
            if s!=-1:
                # This means that a swap is needed
                sorted_vertices[[j,s]]=sorted_vertices[[s,j]]
    
    return sorted_vertices