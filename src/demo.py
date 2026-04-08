import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from render_object import *

# Load the data from the h3.npy file and assign the data to variables.
data=np.load('data/h3.npy',allow_pickle=True)[()]
#print(data)

# Input triangles
verts=data['verts']
vertex_colors=data['vertex_colors']
face_indices=data['face_indices']

# Camera plane and image pixel coordinate system dimensions
# Camera plane
H=data['H']
W=data['W']
# Image pixel coordinate system dimensions
N=data['N']
M=data['M']

# Camera's position and characteristics
cam_eye=data['cam_eye']
#cam_eye=np.reshape(cam_eye,(3,))
cam_up=data['cam_up']
#cam_up=np.reshape(cam_up,(3,))
cam_lookat=data['cam_lookat']
#cam_lookat=np.reshape(cam_lookat,(3,))
focal=data['focal']

# Information about illumination of the object
# Coefficients for the models of ambient light, diffuse reflection and specular reflection
ka=data['ka']
kd=data['kd']
ks=data['ks']
# Phong constant for the specular reflection model
n=data['n']
# Three point light sources and the light intensities of the light radiation emitted from each of them
light_positions=np.array( data['light_positions'] )
light_intensities=np.array( data['light_intensities'] )
# Intensity of the light radiation of ambient light
Ia=data['Ia']
# Background color
bg_color=data['bg_color']

# We will now create different images of the same object setting different conditions regarding lighting from one or more
# point light sources, the type of lighting and the way the object is colored.

# 1. __ Only the first point light source which is located at the point (0,0,5) lights the object __
# 1.1.1. We consider only ambient light and gouraoud shading
shader="gouraud"
l_pos=np.zeros(3)
l_int=np.zeros(3)
l_amb=Ia
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka,
               kd, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/first_light_source_ambient_gouraud.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 1.1.2. We consider only ambient light and phong shading
shader="phong"
l_pos=np.zeros(3)
l_int=np.zeros(3)
l_amb=Ia
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka,
               kd, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/first_light_source_ambient_phong.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 1.2.1 We consider only diffuse reflection and gouraoud shading
shader="gouraud"
l_pos=light_positions[0,:]
l_int=light_intensities[0,:]
l_amb=np.zeros(3)
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, 0,
               kd, 0, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/first_light_source_diffuse_reflection_gouraud.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 1.2.2 We consider only diffuse reflection and phong shading
shader="phong"
l_pos=light_positions[0,:]
l_int=light_intensities[0,:]
l_amb=np.zeros(3)
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, 0,
               kd, 0, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/first_light_source_diffuse_reflection_phong.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 1.3.1 We consider only specular reflection and gouraoud shading
shader="gouraud"
l_pos=light_positions[0,:]
l_int=light_intensities[0,:]
l_amb=np.zeros(3)
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, 0,
               0, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/first_light_source_specular_reflection_gouraud.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 1.3.2 We consider only specular reflection and phong shading
shader="phong"
l_pos=light_positions[0,:]
l_int=light_intensities[0,:]
l_amb=np.zeros(3)
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, 0,
               0, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/first_light_source_specular_reflection_phong.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 1.4.1 We consider overall light model and gouraoud shading
shader="gouraud"
l_pos=light_positions[0,:]
l_int=light_intensities[0,:]
l_amb=Ia
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka,
               kd, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/first_light_source_overall_light_gouraud.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 1.4.2 We consider overall light model and phong shading
shader="phong"
l_pos=light_positions[0,:]
l_int=light_intensities[0,:]
l_amb=Ia
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka,
               kd, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/first_light_source_overall_light_phong.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 2. __ Only the second point light source which is located at the point (-7,0,0) lights the object __
# We consider overall light model and gouraud shading
shader="gouraud"
l_pos=light_positions[1,:]
l_int=light_intensities[1,:]
l_amb=Ia
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka,
               kd, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/second_light_source_overall_light_gouraud.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 3. __ Only the third point light source which is located at the point (0,5,0) lights the object __
# We consider overall light model and gouraud shading
shader="gouraud"
l_pos=light_positions[2,:]
l_int=light_intensities[2,:]
l_amb=Ia
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka,
               kd, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/third_light_source_overall_light_gouraud.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)

# 4. __ All point light sources light the object __
# We consider overall light model and gouraud shading
shader="gouraud"
l_pos=light_positions
l_int=light_intensities
l_amb=Ia
object_image=render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka,
               kd, ks, n, l_pos, l_int, l_amb)
plt.figure(1)
fig1=plt.imshow(object_image)
plt.show()

# Save the image from demo.py
image_path='outputs/all_light_sources_overall_light_gouraud.png'
object_image=cv2.convertScaleAbs(object_image, alpha=(255.0))
object_image=cv2.resize(object_image, (800,800))
isDone=cv2.imwrite(image_path,object_image)
