# DeepDance: Music-to-Dance Motion Choreography with Adversarial Learning
This reop contains training code of paper on Music2Dance generation: "[DeepDance: Music-to-Dance Motion Choreography with Adversarial Learning](https://ieeexplore.ieee.org/abstract/document/9042236/)". [Project Page](http://zju-capg.org/research_en_music_deepdance.html)

## Requirements
- A CUDA compatible GPU
- Ubuntu >= 14.04

## Usage

Download this repo on your computer and create a new enviroment using commands as follows:
 ```
 git clone https://github.com/computer-animation-perception-group/DeepDance_train.git
 conda create -n music_dance python==3.5
 pip install -r requirement.txt
 ```
 Download the processed training data ([fold_json](https://drive.google.com/file/d/18YhFlqkwU6akfjSBgcmywJu_BtfjmAZz/view?usp=sharing), [motion_feature](https://drive.google.com/file/d/18Hk5jEW8DV_AXzWZcvdLkUvlTiVdZ0Sp/view?usp=sharing) and [music_feature](https://drive.google.com/file/d/1VMt_fhG2livx1keh9o9Vu6zwwZPgB3ZZ/view?usp=sharing)), extract and put them under "./dataset", and run the following scripts:
 ```
 bash pretrain.sh
 bash finetune.sh
 ```
 Once the training completed, you can generate novel dances with the training models using our [demo code](https://github.com/computer-animation-perception-group/DeepDance)

## License
Licensed under an GPL v3.0 License and only for research purpose.

## Bibtex
```
@article{sun2020deepdance,
  author={G. {Sun} and Y. {Wong} and Z. {Cheng} and M. S. {Kankanhalli} and W. {Geng} and X. {Li}},
  journal={IEEE Transactions on Multimedia}, 
  title={DeepDance: Music-to-Dance Motion Choreography with Adversarial Learning}, 
  year={2021},
  volume={23},
  number={},
  pages={497-509},}
```
