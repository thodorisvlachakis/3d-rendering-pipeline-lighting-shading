# 3D Rendering: Lighting, Shading & Viewing Pipeline

This repository contains the implementation of the third assignment of the Computer Graphics course.  
The goal of this assignment is to extend a basic rendering pipeline by introducing lighting models and shading techniques in order to produce more realistic images of a 3D scene.

Some components of the rendering pipeline (e.g. triangle rasterization and projection utilities) were developed in previous assignments and are reused and extended in this project.
See also:
- [Triangle Rasterization](https://github.com/thodorisvlachakis/triangle-rasterization-and-shading)  
- [Transformations and Projections](https://github.com/thodorisvlachakis/transformations-and-perspective-projection)

---

## 📌 Overview

This project focuses on extending a basic rendering pipeline to support realistic illumination and shading in a 3D scene.

It simulates the process of capturing images of a 3D scene using a virtual camera, incorporating:

- Lighting models (ambient, diffuse, specular)
- Surface material properties (Phong model)
- Shading techniques (Gouraud and Phong shading)
- Full rendering pipeline from 3D world coordinates to 2D image

The implementation builds upon previous work on:
- Triangle rasterization
- Perspective projection
- Camera transformations

and integrates them into a complete image synthesis framework.

---

## ✨ Features

- Full 3D → 2D rendering pipeline
- Support for multiple point light sources
- Implementation of the **Phong illumination model**
- Two shading techniques:
  - Gouraud shading
  - Phong shading
- Per-vertex normal computation
- Triangle-based rendering using scanline rasterization
- Support for:
  - Ambient lighting
  - Diffuse reflection (Lambertian)
  - Specular reflection (Phong)
- Rendering under different lighting configurations

---

## 🧭 Rendering Pipeline

The rendering process follows these steps:

1. **Normal Computation**
   - Surface normals are computed per vertex based on triangle geometry.

2. **Camera Transformation**
   - World coordinates are transformed into the camera coordinate system.

3. **Perspective Projection**
   - 3D points are projected onto the camera plane.

4. **Rasterization**
   - Projected triangles are converted into pixel coordinates.

5. **Shading**
   - Each triangle is colored using either:
     - Gouraud shading (interpolated vertex intensities)
     - Phong shading (interpolated normals + per-pixel lighting)

6. **Image Composition**
   - Triangles are rendered in depth order to produce the final image.

---

## 💡 Lighting Model

The project implements the **Phong illumination model**, combining:

- **Ambient lighting**  
  Uniform background illumination

- **Diffuse reflection (Lambertian)**  
  Depends on the angle between light direction and surface normal

- **Specular reflection**  
  Depends on viewer position and surface shininess

The final color is computed as the sum of contributions from:

- Multiple point light sources
- Ambient light

---

## 🎨 Shading Techniques

### Gouraud Shading

- Lighting is computed at triangle vertices
- Colors are interpolated across the triangle
- Efficient but may miss specular highlights

### Phong Shading

- Normals are interpolated across the surface
- Lighting is computed per pixel
- Produces smoother and more realistic results

---

## 🛠️ Technologies

- Python 3
- NumPy
- Matplotlib
- OpenCV (cv2)

---

## 📂 Project Structure

```
3d-rendering-pipeline-lighting-shading
│
├── src/                    # Python source code
│ ├── calculate_normals.py
│ ├── compute_lines_triangle.py
│ ├── demo.py
│ ├── light.py
│ ├── lookat.py
│ ├── perspective_project.py
│ ├── rasterize.py
│ ├── render_img.py
│ ├── render_object.py
│ ├── shade_gouraud.py
│ ├── shade_phong.py
│ ├── sort_vertices.py
│ ├── TransformClass.py
│ ├── vector_interp.py
│ └── world2view.py
│
├── data/                   # Input data
│ └── hw3.npy
│
├── outputs/                # Generated images
│ ├── All_light_sources_gouraud.png
│ ├── All_light_sources_phong.png
│ ├── First_light_source_ambient_gouraud.png
│ ├── First_light_source_ambient_phong.png
│ ├── First_light_source_diffuseReflection_gouraud.png
│ ├── First_light_source_diffuseReflection_phong.png
│ ├── First_light_source_overall_light_model_gouraud.png
│ ├── First_light_source_overall_light_model_phong.png
│ ├── First_light_source_specularReflection_gouraud.png
│ ├── First_light_source_specularReflection_phong.png
│ ├── Second_light_source_overall_light_model_gouraud.png
│ ├── Second_light_source_overall_light_model_phong.png
│ ├── Third_light_source_overall_light_model_gouraud.png
│ └── Third_light_source_overall_light_model_phong.png
│
├── docs/                   # Documentation
│ ├── hw3_2024.pdf
│ └── report.pdf
│
├── README.md
└── .gitignore

```
---

## ⚙️ How to Run

### 1. Requirements

Make sure you have the following installed:

- Python 3.x  
- NumPy  
- Matplotlib  
- OpenCV  

You can install the required libraries using:

```bash
pip install numpy matplotlib opencv-python

```

### 2. Run the demos

Navigate to the src/ folder and run:
    _python demo.py_

The demo scripts load the required input data internally and generate the final rendered images.
The required input data is provided in the data/ folder.

---

## 🧩 Key Components

- `render_object.py`: Main rendering pipeline combining projection, rasterization, and shading  
- `light.py`: Implementation of the Phong illumination model  
- `calculate_normals.py`: Computes per-vertex normals  
- `shade_gouraud.py`: Vertex-based shading with color interpolation  
- `shade_phong.py`: Per-pixel shading using interpolated normals  

---

## 🖼️ Results

The project generates multiple rendered images demonstrating:

- Different light sources (3 individual + all combined)
- Different lighting components:
  - Ambient only
  - Diffuse only
  - Specular only
  - Full lighting model
- Comparison between Gouraud and Phong shading

Total outputs include:
- Multiple configurations per shading method
- Side-by-side qualitative comparison of rendering techniques

---

## 🧠 Key Concepts

- Phong illumination model  
- Gouraud vs Phong shading  
- Surface normals and lighting calculations  
- Perspective projection and camera modeling  
- Triangle rasterization

---

## 📄 Notes

- The implementation assumes no distance attenuation for lighting.
- Only triangles fully inside the camera view are rendered.
- OpenCV (cv2) is used for image post-processing (scaling, image resizing and color format conversion) before saving the final output.
- An additional assignment component related to texture mapping is included in the provided description (hw3_2024.pdf) but is not implemented in this project.