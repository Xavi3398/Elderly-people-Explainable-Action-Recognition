# Elderly-people-Explainable-Action-Recognition
This project contains all code used to train TimeSformer on video action recognition task using ETRI-Activity3D dataset, 
as a Master's Thesis for the Master’s Degree in Intelligent Systems (MUSI) at the Universitat de les Illes Balears (UIB).
You can find the trained model along with traininig and evaluation logs, generated explanations and more [here](https://drive.google.com/drive/folders/1Cn107VogSNAHN03PTPMW-Y4SCKoqP0KC?usp=sharing)

## Installation
This project has dependencies on the [MMAction2](https://github.com/open-mmlab/mmaction2) and [LIME](https://github.com/marcotcr/lime) projects. 
Please, install them before trying to execute any notebook, and then add the mmaction2 and the lime folders to their installation root, so as to add some added or modified files.
The [ETRI-Activity3D](https://ai4robot.github.io/etri-activity3d-en/#) dataset is also required to run the training and some scripts.

## Explanations in space and time
<img src="/images/xy-different-samples-and-x-plus-y-and-time.png" width="900" /> 

## Explanation examples
<img src="/images/best-explanations.pn" width="900" /> 
