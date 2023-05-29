![mlops_banner](https://github.com/Bismuth-Consultancy-BV/MLOPs/assets/115803478/abc396a5-be3c-4bee-bdb9-a3721a829dcd)


# Houdini MLOPs 2.0
Free and Open Source Machine Learning Plugin for Houdini developed by Ambrosiussen Holding and [Entagma](https://entagma.com/), Licensed and Distributed by [Bismuth Consultancy B.V.](https://www.bismuthconsultancy.com/) _By downloading or using the plugin (or any of its contents), you are agreeing to the LICENSE found in this repository and [Terms of Service of Bismuth Consultancy B.V.](https://www.bismuthconsultancy.com/s/EN_Terms_And_Conditions-f5sk.pdf)_

| Paul Ambrosiussen | Entagma | Discord |
| --------------- | --------------- | --------------- |
| [![](https://img.shields.io/badge/twitter-%230077B5.svg?style=for-the-badge&logo=twitter)](https://twitter.com/ambrosiussen_p) | [![](https://img.shields.io/badge/twitter-%230077B5.svg?style=for-the-badge&logo=twitter)](https://twitter.com/entagma) | [![](https://img.shields.io/badge/discord-%230077B5.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/rKr5SNZJtM) |
| [![](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/paulambrosiussen/) |



## Promo Video
<a href="http://www.youtube.com/watch?feature=player_embedded&v=izLicMTBYUg
" target="_blank"><img src="http://img.youtube.com/vi/izLicMTBYUg/0.jpg" 
alt="2.0 Release Promo" width="640" height="480" border="0" /></a>


# Installing for Houdini
To install the plugin for the first time, follow these steps:
1. Clone this repository and make note of the directory you have cloned it to.
2. Copy the `MLOPs.json` file found in the repository root, and paste it in the $HOUDINI_USER_PREF_DIR/packages/ folder.
3. Edit the `MLOPs.json` file you just pasted, and modify the `$MLOPS` path found inside. Set the path to where you cloned the repository to in step one.
4. Install `git`. Follow the instructions for your relevant OS [here](https://github.com/git-guides/install-git).
5. Launch Houdini and open the `MLOPs` shelf. Click the `Install Dependencies` shelf button. Restart Houdini once complete.
6. After restarting Houdini, open the `MLOPs` shelf. Click the `Download Model` button. Optionally change the `Model Name` parameter to a custom model, or just leave as is and hit `Download` to work with the default Stable Diffusion Model.
7. In the MLOPs nodes, use the dropdown on the `[type] Model` parameters to select a downloaded model to use. You can also provide a repo name from the [Huggingface Library](https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads), and the nodes will download it for you. For example `runwayml/stable-diffusion-v1-5`.


# Downloading Models
- By default, `$MLOPS_SD_MODEL` is the path to a SINGLE model used by all Stable Diffusion nodes by default. You can set this to be your preferred default model.
- By default, the plugin will cache all downloaded models to the folder specified by `$MLOPS_MODELS`. (Notice the S at the end) This will make them show up in the dropdowns for the model paths on the nodes.
Both of the above varibles can be changed in the `MLOPS.json` to suit your preference.

# Troubleshooting
- If you get an error saying "Torch not compiled with CUDA enabled". Uninstall pytorch in your system python, restart your PC and hit the `Install Dependencies` shelf button again. 
- Metal users should set the compute device on MLOPs nodes to "MPS", and set the following environment variable for it to work: `PYTORCH_ENABLE_MPS_FALLBACK=1`
- If you get an error with "Could not load library cudnn_cnn_infer64_8.dll" and you have Octane installed as a plugin for Houdini, try disabling it and restart Houdini.
- If you get an error similar to: "Unexpected self.size(-1) must be divisible by 4 to view Byte as Float (different element sizes), but got 2683502, <class 'RuntimeError'>". Try deleting the model cache in `$MLOPS_MODELS/cache/[your model]` and try again. This is likely caused by a corrupt cache.
- Other plugins we know cause issues installing MLOPS dependencies: Renderman, Octane. Disable these while installing MLOPs and its dependencies. After installing MLOPs and its dependencies you can re-enable them.
- If you get strange Python errors and you have tried several things already, make sure you dont have a conflicting `PYTHONPATH` environment variable set. If that is the case remove it and restart Houdini (And the Launcher if you use it)

# Notes
- We have provided a basic example file in this repo. You can find it in the `hip/` folder.
- This plugin installs quite a few dependencies. You can find them in `requirements.txt`.
- Digital Assets (HDAs) are stored and distributed in the expanded format. You can use [hotl](https://www.sidefx.com/docs/houdini/ref/utils/hotl.html) to collapse them if need be.
