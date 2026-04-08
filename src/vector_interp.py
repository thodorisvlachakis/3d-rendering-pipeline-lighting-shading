import numpy as np
# This function calculates the "vector value" V corresponding to a position "p" by doing
# linear interpolation between "vector values" V1 and V2 which corresponds to positions p1 and p2, respectively.
# The point p=(x,y) is assumed to belong to the line segment defined by p1=(x1,y1) and p2=(x2,y2).
# This function takes as input the x-coordinate or the y-coordinate of the point p (this input is the "coord").
# Acoording to this coordinate, the function calculates the other coordinate and so defines p as weighted sum of p1 and p2.
# Since p belongs to the line segment defined by p1 and p2, it can be written as p= λ*p1 + (1-λ)*p2, where λ∈[0,1].
# So, the main goal is to calculate the value of λ, which shows the exact position of p on the line segment p1-p2.
# We can easily find the value of λ, if we know the y-coordinate of p, by using Thali's Theorem as λ=|p2p| / |p1p2|,
# where |c1c2| is the measure of vector with start to c1 and end to c2, in general.
# Then, function uses this value of λ in order to calculate V as weighted sum of V1 and V2, i.e V= λ*V1 + (1-λ)*V2.

# INPUTS:
# p1 : a vector of length 2 which describes the two-dimensional coordinates of the point p1 where V1 corresponds.
# p2 : a vector of length 2 which describes the two-dimensional coordinates of the point p2 where V2 corresponds.
# V1 : a vector of unknown length which describes the vector values that corresponds to position p1.
# V2 : a vector of unknown length which describes the vector values that corresponds to position p2.
# dim : an integer. Valid values are 1 and 2.
# coord: an integer which describes either the x-coordinate of the point p ,if dim=1, or the y-coordinate of the point
#        p, if dim=2.
# OUTPUTS:
# V: function returns V which is a vector of unknown length which describes the vector values that corresponds to position p.
  
def vector_interp(p1,p2,V1,V2,coord,dim):
    # Special cases:
    if (dim!=1 and dim!=2):
        return "Invalid value for the input with name: dim"
    if (p1==p2).all():
        # This means that line segment defined by p1 and p2 is not a line segment but a point
        V=V1
        return V

    #General cases:

    if dim==2:
        # y-coordinate of p is given.
        # calculate lamda=λ by using Thali's Theorem.
        if p1[1]!=p2[1]:
            lamda=abs( ( coord-p2[1] ) / ( p2[1]-p1[1] ) )
        else:
            # This means y1=y2 which result in horizontal line segment and so we have to calculate lamda from x-coordinate 
            return "Giving the y-dimension of p is not useful because the points p1 and p2 are in the same horizontal line."
    elif dim==1:
        # x-coordinate of p is given.
        # calculate lamda=λ by using Thali's Theorem.
        if p1[0]!=p2[0]:
            if p1[1]==p2[1]:
                # This means y1=y2 which result in horizontal line segment and so we have to calculate lamda from x-coordinate
                lamda=abs( ( coord-p2[0] ) / ( p2[0]-p1[0] ) )
            else:
                # y-coordinate of p must be calculated
                # find the line y=m*x + b defined by p1 and p2
                m=(p2[1]-p1[1]) / (p2[0]-p1[0])
                b=p1[1]-m*p1[0]

                # y-coordinate of p
                y=m*coord + b
                lamda=abs( ( y-p2[1] ) / ( p2[1]-p1[1] ) )
        else:
            # This means x1=x2 which result in vertical line segment and so we have to calculate lamda from x-coordinate 
            return "Giving the x-dimension of p is not useful because the points p1 and p2 are in the same vertical line."    
    
    V=lamda*np.asarray(V1) + (1-lamda)* np.asarray(V2)

    return V