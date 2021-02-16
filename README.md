## Pytorch implementation of Learning-based Video Motion Magnification
### 2021 / 02 / 16 / winter_internship

Most of the source code was referenced and copied in these materials.
1. https://github.com/12dmodel/deep_motion_mag
2. https://github.com/Fangyh09/Image2LMDB
3. https://pytorch.org/tutorials/

Caution!
Some code has an inefficient structure. 
Modify and use the code according to the purpose.

## Installation
    conda create -n env_name python==3.6.9
    pip install ffmpeg==1.4
    pip install -r requirements.txt
This code has been tested with torch 1.7.1, torchvision 0.8.2, CUDA 10.2, conda 4.6.9, python 3.6.9, Ubuntu 16.04.

## Dataset
    ├── main.py
    ├── train
    │   ├── 1
    │   │   ├── amplified
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   ├── .
    │   │   │   └── .
    │   ├── 2   
    │   │   ├── frameA
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   ├── .
    │   │   │   └── .
    │   ├── 3   
    │   │   ├── frameB
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   ├── .
    │   │   │   └── .
    │   ├── 4   
    │   │   ├── frameC
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   ├── .
    │   │   │   └── .
    ├── README.md
    ├── requirements.txt
    ├── .

## Inference
This command is executed in dynamic mode. Delete "--velocity_mag" for static mode.

    python main.py --phase="play" --checkpoint_path="Path to the model.tar" --vid_dir="Path to the directory where the video frames are located" 
    --out_dir="path to the output" --velocity_mag

**Inference with temporal filtered**

This code supports two types of <filter_type>, {"butter" and "differenceOfIIR"}.

    python main.py --phase="play_temporal" --checkpoint_path="Path to the model.tar" --vid_dir="Path to the directory where the video frames are located" --out_dir="path to the output" --amplification_factor=<amplification_factor> --fl=<low_cutoff> --fh=<high_cutoff> --fs=<sampling_rate> --n_filter_tap=<n_filter_tap> --filter_type=<filter_type>
    
## Train
**PNG Image Dataset to lmdb file**

create the lmdb file in the train dir
        
    python pngtolmdb.py path/to/master/directory

**Train**

    python main.py --phase="train" --checkpoint_path="Path to the model.tar" --data_path="Path to the directory where the lmdb file are located"

    
## Citation
    @article{oh2018learning,
      title={Learning-based Video Motion Magnification},
      author={Oh, Tae-Hyun and Jaroensri, Ronnachai and Kim, Changil and Elgharib, Mohamed and Durand, Fr{\'e}do and Freeman, William T and Matusik, Wojciech},
      journal={arXiv preprint arXiv:1804.02684},
      year={2018}
    }
