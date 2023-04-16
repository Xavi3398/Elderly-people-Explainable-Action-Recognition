# Elderly-people-Explainable-Action-Recognition
This project contains all code used to train TimeSformer, TANet and TPN on the video activity recognition task using the ETRI-Activity3D dataset.
You can find the trained model along with traininig and evaluation logs and generated explanations [here](https://drive.google.com/drive/folders/1Cn107VogSNAHN03PTPMW-Y4SCKoqP0KC?usp=share_link).

## Installation
This project has dependencies on the [MMAction2](https://github.com/open-mmlab/mmaction2) and [LIME](https://github.com/marcotcr/lime) projects. 
Please, install them before trying to execute any notebook, and then add the mmaction2 and the lime folders to their installation root, so as to add some added or modified files.
The [ETRI-Activity3D](https://ai4robot.github.io/etri-activity3d-en/#) dataset is also required to run the training and some scripts.

## Explanations in space and time
<img src="/images/xy-different-samples and x plus y and time.png" width="900" /> 

## Explanation examples
<img src="/images/best explanations.png" width="900" /> 
