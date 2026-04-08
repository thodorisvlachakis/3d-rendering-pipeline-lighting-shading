import numpy as np

# This function implements the overall lighting model for a 3D point. In fact, the function calculates the intensity of the
# light radiation received and reflected by a specific point that belongs to a surface (that is made of some Phong material) in
# 3D space. The calculation refers to the total intensity of the light radiation received by the point due to, simultaneously,
# diffuse ambient light, diffuse reflection and specular reflection. To construct this overall illumination model, the
# diffuse ambient light model, the diffuse-Lambertian reflection model and the specular reflection model are used seperately
# one-to-one) and then the results are combined to end up the received and reflected light radiation by the 3D point of the
# surface. When we refer to the total intensity of the light radiation, we mean the intensity of the trichromatic radiation
# that respects to the point of interest. This means that the function considers the three-dimensional color vector (RGB vector)
# which describes the color of the point and applies the above three models of illumination (combining the results for the light
# intensity, as described) to each component of the color vector. So, the function calculates the light intensity that respects
# to each chromatic component, i.e to each wave length of the emmited light radiation thst can be received and reflected by the
# specified point of the suface. Then the composition of a three-dimensional vector that desctribes the intensity of illumination
# of each component of the RGB vector, i.e the color, of the point can be made. This is the trichromatic radiation. 
# Furthermore, analyzing the way that this function follows, in order to result in trichromatic radiation (denoted by I), we mark
# how the three models (ambient light, diffuse reflection, specular reflection) work. Firstly, the ambient light model is a simple
# illumination model and describes the case of illumination of a point by means of light that is diffused inthe environment 
# around the surface to which the point belongs. This light model does not need the coordinates of any point light source to
# calculate the intensity reflected by the point of surface, because it refers to diffused light from the environment. Secondly,
# the diffuse-Lambertian reflection model describes the case of the diffuse reflection (in the invironment around the surface)
# of the received light radiation by the specified point of the surface. In this case, the position of the point light source
# illuminating the point of interest is important. Since the reflection is diffuse, the position of the observer does not matter
# in terms of the reflected light radiation received by him. Thirdly, the specular reflection model describes the case of specular
# reflection of light radiation from a specific point and calulates the intensity of this reflected radiation received by an
# observer. In this case, therefore, the position of the observer is important, as the intensity of the received radiation 
# depends on the viewing angle. In addition, the intensity of the reflected light radiation by the point depends on the anlge
# of incidence (as in the Lambertian reflection model), so the position of the point light source is also needed. This function
# simulates all these models and in the end combine the results. The combination refers to simple sum for each component of the
# trichromatic radiation calculated using each model. Generalizing, the function calculates the intensity of light radiation,
# due to diffuse ambient light, diffuse reflection and specular reflection, when the light radiation received and reflected by
# the point of interest is due to N>=1 point light sources. In order to implement that model, the function calculates the total
# illumination model for each source separately, as described above, and finally composites the results, considering that each
# computed trichromatic vector contributes cumulatively to the total trichromatic vector I that the function returns.     
# ** We note that the attenuation due to distance is omitted in terms of diffuse reflection and specular reflection models.

# INPUTS:
# point: a 3×1 column vector which includes the 3D-coordinates of the point of interest.
# normal: a 3×1 column vector which includes the 3D-coordinates of the normal vector of the surface (i.e the vectical
#         vector on the surface) calculated at the point of interest (whose coordinates are given in the column vector called
#         "point"). The normal vector is towards the outside of the surface, i.e the towards the side of the observer.
# color: a 3×1 column vector which includes the color coordinates (R,G,B) , i.e the RGB color, of the point of 
#        interest.
# cam_pos: a 3×1 column vector which includes the 3D-coordinates of the point where the observer (i.e the camera) is
#          located.
# ka: a float number (it is between 0 and 1) which represents the coefficient of diffuse ambient light.
# kd: a float number (it is between 0 and 1) which represents the coefficient of diffuse reflection.
# ks: a float number (it is between 0 and 1) which represents the coefficient of specular reflection. 
# n: an integer that represents the Phong's constant used.
# l_pos: a N×3 2D-array whose each row includes the 3D-coordinates of a point light source.
# l_int: a N×3 2D-array whose i-th row includes intensity of the light radiation (it is a vector of length 3) that respects to
#        (i.e that is emmitted by) the point light source that corresponds to the i-th row of l_pos array.
# l_amb: a 3×1 vector which includes the components of the intensity of ambient light. Any of components belongs to interval 
#        [0,1]. 
#
# OUTPUTS:
# I: a 3×1 column vector which includes the intensity of the trichromatic light radiation reflected by the point of interest.

def light(point, normal, vcolor, cam_pos, ka, kd, ks, n, l_pos, l_int, l_amb)->np.ndarray:
    I=np.zeros(3)
    if len(l_int.shape)==1:
        # This means 1D array, i.e vector
        N=1
    else:
        N=l_int.shape[0] # N is the number of point light sources that exist to illuminate the point.

    # Assure that normal vector is a unit vector
    normal = normal / np.linalg.norm(normal)

    # Calculate the total intesity of the trichromatic light radiation "I" reflected by the point "point" using the three
    # illumination models: ambient light, diffuse reflection and specular reflection and in the end combining the results of
    # each model cumulatively.

    # First Step: Calculate the intesity of the light radiation reflected by the point, due to ambient light. Consider
    # all the point light sources that illuminate the point.
    Ia = ( l_amb*ka ) * vcolor

    # Seconf Step: Calculate the intensity of the light radiation reflected by the point, due to diffuse reflection. Consider
    # all the point light sources that illuminate the point.  
    # Initialization of the intensity of light radiation reflected by the point, due to diffuse reflection.
    Id=np.zeros(3)
    for i in range(N):
        # Declare the position where the i-th point light source is located.
        if len(l_pos.shape)==1:
            source_pos=l_pos
        else:
            source_pos=l_pos[i,:]
        # Delcare the unit vector L_unit that is defined from the point of interest to the point light source.
        L_unit= (source_pos-point) / np.linalg.norm(source_pos-point)
        
        # Use the diffuse-Labertian reflection model to calculate the contribution of the i-th point light source in total
        # intensity of trichromatic light radiation reflected by the point, due to diffuse reflection
        if np.dot(normal,L_unit)>0:
            if len(l_int.shape)==1:
                Id = Id + ( ( l_int*kd * np.dot(normal,L_unit) ) * vcolor )    
            else:
                Id = Id + ( ( l_int[i,:]*kd * np.dot(normal,L_unit) ) * vcolor )

    # Third Step: Calculate the intensity of the light radiation reflected by the point, due to specular reflection. Consider
    # all the point light sources that illuminate the point.  
    # Initialization of the intensity of light radiation reflected by the point, due to specular reflection.
    Is=np.zeros(3)
    for i in range(N):
        # Declare the position where the i-th point light source is located.
        if len(l_pos.shape)==1:
            source_pos=l_pos
        else:
            source_pos=l_pos[i,:]
        # Declare the unit vector L_unit that is defined from the point of interest to the point light source.
        L_unit= (source_pos-point) / np.linalg.norm(source_pos-point)
        
        # Declare the unit vector V_unit that is defined from the point of interest to the point where the observer (i.e the
        # camera) is located.
        V_unit= (point-cam_pos) / np.linalg.norm(point-cam_pos)
        
        # Declare the unit vector R_unit that defines the direction of the line of the perfect specular reflection, as declared
        # in specular reflection model
        R_unit= ( 2*normal* np.dot(normal,L_unit) ) - L_unit

        # Use the specular reflection model to calculate the contribution of the i-th point light source in total intensity of
        # trichromatic light radiation reflected by the point and received by the observer (i.e camera), due to specular reflection.
        if (np.dot(normal,L_unit)>=0 and np.dot(R_unit,V_unit)>=0):
            if len(l_int.shape)==1:
                Is = Is + ( l_int* ks* ( (np.dot(R_unit,V_unit))**n ) * vcolor)
            else:
                Is = Is + ( l_int[i,:]* ks* ( (np.dot(R_unit,V_unit))**n ) * vcolor)
    
    # Final Step: Define the overall illumination model combining the results of the partial models.
    I = Ia + Id + Is
    for i in range(3):
        if I[i]>1:
            I[i]=1

    return I