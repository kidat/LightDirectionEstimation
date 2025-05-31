# Light Direction Estimation from Depth (MoGe) and Shadow (SSIS)

This project leverages MoGe for depth estimation and SSIS for object-shadow association to accurately estimate the direction of the light source present at the time of image capture.
Such light direction estimation can be valuable for applications including realistic scene relighting, photo editing, and inferring the time of day based on lighting conditions in the image.

To run this code, clone all of the following Git repositories and install the required data and dependencies for each project, following the instructions provided on their official websites.
MoGe:   git clone https://github.com/microsoft/MoGe.git
SSIS:   git clone https://github.com/stevewongv/SSIS.git
LightDir: git clone https://github.com/kidat/LightDirectionEstimation.git

      ### Project structure
      ├──MoGe: 
      ├──SSIS: 
            ├──demo
      ├──LightDirectionEstimation
            ├──config
            ├──models
            .....
            ├──main.py
            
#### Before running this project, make sure MoGe and SSIS are correctly installed, and all data and dependencies are setup according to the instructions provided on their official websites. Test them separately.
   
### Demo

To evaluate the results, try the command example:

cd LightDirectionEstimation

pip install -r requirements.txt

python main.py --input ./samples


This project receives a folder of images and generates depth maps, object-shadow associations, light direction estimations in 2D, and 3D environment mappings in .ply format for each image.

#### The result of 5 sample examples are provided in the Result folder
#### All the codes in a single file is also provided in the onefile folder, and it can be run from the SSIS/demo directory.
