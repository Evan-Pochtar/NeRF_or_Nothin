# NeRF or Nothing
#### An optimized Neural Radiance Field for depth calculation.
Created by Evan Pochtar, Evan Krainess, Hady Kotifani, and William Anderson

## Folders and Files
* **/comparison** - Contains all comparison functions used, as well as the error functions used to calculate the final results.
  * **/Stereo** - Contains the Stereo depth mapping function.
  * **/MiDaS** - Contains the MiDaS depth mapping function as well as an example image used to test it's accuracy with transparent images.
* **/Data_Utils** - Contains all functions used to translate the dataset into workable material for the NeRF. This includes blurring, creating NPZ files, ect.
* **/Old_Versions** - This folder contains all no longer used versions of the NeRF. This is included to show the progression of the code from the originalTinyNerf.ipynb, then turning into it's python translated version tinyNerfPy.py, then finally an UpdatedTinynerf.py. This Updated TinyNeRF is the version that was translated to pytorch for our final version.
* **/materials** - Output images from the Old_Versions NeRF models, will be auto created if this folder doesn't exist when running UpdatedTinynerf.py.
* **/torch_materials** - Output images from the torchNeRF model, will be auto created if this folder doesn't exist when running torchNerf.py.
* **/** - Root of the project
  * **.NPZ files** - The main dataset of the NeRF, to add a new dataset place the file into the root directory and change the "npzName" parameter in torchNerf.py
  * **Requirements.txt** - The requirements to run any of the NeRF's within this repository. To install, use the command `pip install -r requirements.txt`.
  * **renderScene.py** - Helper functions to translate image data to useable NeRF data, further data processing functions can be found in the **/Data_Utils** folder.
  * **torchNerf.py** - The main optimized Neural Radience field created and used for this project. Any parameters that can be changed can be found under the `# Hyperparameters` comment in the file. To run this, install all requirements, change all parameters wanted, and run `python3 torchNerf.py`.

## Acknowledgements
We thank the developers of [NeRF](https://github.com/bmild/nerf) and [NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch) for the main codebase behind this project, as well as ideas for the translation of the codebase into PyTorch. We also thank [BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF) for ideas behind the data preprocessing used for the main dataset.