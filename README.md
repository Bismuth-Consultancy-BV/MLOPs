![SideFXLabs logo](https://github.com/Bismuth-Consultancy-BV/MLOPs/blob/main/help/images/mlops_banner.png)

# Houdini MLOPs
Free and Open Source Machine Learning Plugin for Houdini developed by Ambrosiussen Holding and Entagma, Licensed and Distributed by Bismuth Consultancy BV.

Paul Ambrosiussen:

[![](https://img.shields.io/badge/twitter-%230077B5.svg?style=for-the-badge&logo=twitter)](https://twitter.com/ambrosiussen_p)
[![](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/paulambrosiussen/)

Entagma:

[![](https://img.shields.io/badge/twitter-%230077B5.svg?style=for-the-badge&logo=twitter)](https://twitter.com/entagma)

You can join the Discord to chat about the plugin: https://discord.gg/rKr5SNZJtM

By downloading or using the plugin (or any of its contents), you are agreeing to the LICENSE found in this repository and [Terms of Service of Bismuth Consultancy B.V.](https://www.bismuthconsultancy.com/s/EN_Terms_And_Conditions-f5sk.pdf)

Please note that the plugin is currently in beta, and may change at anytime without notice.

# Installing for Houdini
To install the plugin for the first time, follow these steps:
1. Clone this repository and make note of the directory you have cloned it to.
2. Copy the `MLOPs.json` file found in the repository root, and paste it in the $HOUDINI_USER_PREF_DIR/packages/ folder.
3. Edit the `MLOPs.json` file you just pasted, and modify the `$MLOPS` path found inside. Set the path to where you cloned the repository to in step one.
4. Launch Houdini and open the `MLOPs` shelf. Click the `Install Dependencies` shelf button. Restart Houdini once complete.
5. After restarting Houdini, open the `MLOPs` shelf. Click the `Download Model` button. Optionally change the `Model Name` parameter to a custom model, or just leave as is and hit `Download` to work with the default Stable Diffusion Model.
12. In the MLOPs nodes, use the dropdown on the `[type] Model` parameters to select a downloaded model to use. You can also provide a repo name from the [Huggingface Library](https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads), and the nodes will download it for you. For example `runwayml/stable-diffusion-v1-5`.  

# Notes
- We have provided a basic example file in this repo. You can find it in the `hip/` folder.