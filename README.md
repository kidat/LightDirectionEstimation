# Light Direction Estimation from Depth and Shadow

This project leverages MoGe for depth estimation and SSIS for object-shadow association to accurately estimate the direction of the light source present at the time of image capture.
Such light direction estimation can be valuable for applications including realistic scene relighting, photo editing, and inferring the time of day based on lighting conditions in the image.

To run this code, clone all of the following Git repositories and install the required data and dependencies for each project, following the instructions provided on their official websites.
MoGe:   git clone https://github.com/microsoft/MoGe.git
SSIS:   git clone https://github.com/stevewongv/SSIS.git
LightDir: git clone https://github.com/kidat/LightDirectionEstimation.git

      ### Project structure
      ├──MoGe: 
      ├──SSIS: 
      ├──LightDirectionEstimation
            ├──config
            ├──models
            .....
            ├──main.py
            
      
Demo

To evaluate the results, try the command example:

cd LightDirectionEstimation

python main.py --input ./samples


This project receives a folder of images and generates depth maps, object-shadow associations, light direction estimations in 2D, and 3D environment mappings in .ply format for each image.
