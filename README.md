# EMP-Net
The Official implementation for [Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00305.pdf) (ECCV 2024).

- [Prerequisite](#Prerequisite)
- [Data](#Data)
- [Running](#Running&Testing)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)
- [Contact](#Contact)


<a name="Prerequisite"></a>

# Prerequisite

Requirements:
- Python>=3.6
- torch>=1.5
- torchvision (version corresponding with torch)
- simplejson==3.11.1
- decord>=0.6.0
- pyyaml
- einops
- oss2
- psutil
- tqdm
- pandas

optional requirements
- fvcore (for flops calculation)

You can create environments simply with the following command:
```
conda env create -f environment.yaml
```
 
<a name="Data"></a>

# Data

First, you need to download the datasets from their original source and put them into [data](./data/):

- [SSV2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
- [Kinetics400](https://github.com/Showmax/kinetics-downloader)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

Then, prepare data according to the [splits](configs/projects/EMP_Net/) we provide.


### Preprocessing Something-Something-V2 dataset

Following the setup of [pytorch-video-understanding](https://github.com/alibaba-mmai-research/TAdaConv) codebase, the video decoder [decord](https://github.com/dmlc/decord) has difficulty in decoding the original webm files. So we provide a script for preprocessing the `.webm` files in the original something-something-v2 dataset to `.mp4` files. To do this, simply run:
```
python datasets/utils/preprocess_ssv2_annos.py --anno --anno_path path_to_your_annotation
python datasets/utils/preprocess_ssv2_annos.py --data --data_path path_to_your_ssv2_videos --data_out_path path_to_put_output_videos
```

Remember to make sure the annotation files are organized as follows:
```
-- path_to_your_annotation
    -- something-something-v2-train.json
    -- something-something-v2-validation.json
    -- something-something-v2-labels.json
```


<a name="Running"></a>

# Running

The entry file for all the runs are [runs](runs/run.py). 

Before running, some settings need to be done. 

For an example run of 1 shot on SSV2-Small, all configurations can be found from [configs](configs/projects/EMP_Net/ssv2_small/EMP_Net_SSv2_Small_1shot_v1.yaml), you can modify any configurations as needed.

Then the codebase can be run by:

```
python runs/run.py --cfg ./configs/projects/EMP_Net/ssv2_small/EMP_Net_SSv2_Small_1shot_v1.yaml
```

<a name="Citation"></a>

# Citation
If you find this model useful for your research, please use the following BibTeX entry.
```
@inproceedings{wuefficient,
  author={Wu, Cong and Wu, Xiao-Jun and Li, Linze and Xu, Tianyang and Feng, Zhenhua and Kittler, Josef},
  title={Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```

<a name="Acknowledgement"></a>

# Acknowledgement
Thanks for the framework provided by [CLIP-FSAR](https://github.com/alibaba-mmai-research/CLIP-FSAR), 
which is source code of the published work [CLIP-guided Prototype Modulating for Few-shot Action Recognition](https://arxiv.org/pdf/2303.02982) in IJCV 2023. 


<a name="Contact"></a>

# Contact
For any further questions, feel free to contact: `congwu@jiangnan.edu.cn`.