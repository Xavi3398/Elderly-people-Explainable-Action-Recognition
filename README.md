# Elderly-people-Explainable-Action-Recognition
This project contains all code used in the conference article [*"Explainable activity recognition for the elderly"*](https://doi.org/10.1145/3612783.3612790). It included training of TimeSformer, TANet and TPN networks on the human activity recognition from video task using the ETRI-Activity3D dataset, and the generation of separated space and time explanations.

You can find the trained models along with traininig and evaluation logs and generated explanations [here](https://drive.google.com/drive/folders/1Cn107VogSNAHN03PTPMW-Y4SCKoqP0KC?usp=share_link).

## Installation
This project has dependencies on the [MMAction2](https://github.com/open-mmlab/mmaction2) and [LIME](https://github.com/marcotcr/lime) projects. Please, install them before trying to execute any notebook, and then add the mmaction2 and the lime folders to their installation root, so as to add some added or modified files.

The [ETRI-Activity3D](https://ai4robot.github.io/etri-activity3d-en/#) dataset is also required to run the training and some scripts.

## Explanations in space and time
<img src="/images/xy-different-samples and x plus y and time.png" width="900" /> 

## Explanation examples
<img src="/images/best explanations.png" width="900" /> 

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Grant PID2019-104829RA-I00 funded by MCIN/ AEI /10.13039/501100011033. Project EXPLainable Artificial INtelligence systems for health and well-beING (EXPLAINING)

F. X. Gaya-Morey was supported by an FPU scholarship from the Ministry of European Funds, University and Culture of the Government of the Balearic Islands.

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{gaya_morey2024explainable,
	title        = {Explainable activity recognition for the elderly},
	author       = {Gaya-Morey, F. Xavier and Manresa-Yee, Cristina and Buades-Rubio, Jose M.},
	year         = 2024,
	booktitle    = {Proceedings of the XXIII International Conference on Human Computer Interaction},
	location     = {Lleida, Spain},
	publisher    = {Association for Computing Machinery},
	address      = {New York, NY, USA},
	series       = {Interacci\'{o}n '23},
	doi          = {10.1145/3612783.3612790},
	isbn         = 9798400707902,
	articleno    = 6,
	numpages     = 8
}

```
