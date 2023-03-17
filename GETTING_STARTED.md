## Installation

First, clone the repository locally:

```bash
git clone https://github.com/kuantuna/odvis.git
cd odvis
```

### Requirements
- Linux or macOS with Python ≥ 3.6

- PyTorch ≥ 1.9.0 and torchvision that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org/get-started/locally/)

- OpenCV is optional and needed by demo and visualization


Install and build detectron2 with the guidelines from [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

```bash
pip install timm
pip install shapely
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```

## Data Preparation



Download and extract 2019 version of YoutubeVIS train and val images with annotations from [CodeLab](https://competitions.codalab.org/competitions/20128#participate-get_data) or [YouTubeVIS](https://youtube-vos.org/dataset/vis/), download [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate)  and COCO 2017 datasets. Then, link datasets:

```bash
mkdir datasets
cd datasets
ln -s /path_to_coco_dataset coco
ln -s /path_to_YTVIS19_dataset ytvis_2019
ln -s /path_to_ovis_dataset ovis
```



Extract YouTube-VIS 2019, OVIS, COCO 2017 datasets, we expect the directory structure to be the following:

```
odvis
├── datasets
│   ├──ytvis_2019
│   ├──ovis 
│   ├──coco 
...
ytvis_2019
├── train
├── val
├── annotations
│   ├── instances_train_sub.json
│   ├── instances_val_sub.json
...
ovis
├── train
├── valid
├── annotations_train.json
├── annotations_valid.json
...
coco
├── train2017
├── val2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
```


## Downloading Pretrained Model Weights

```bash
mkdir models
cd models
```

Download and save the model you need into the models folder. Afterwards, edit the MODEL.WEIGHTS in the corresponding yaml config file.

| Backbone   | Phase 1 | Phase 2 | Phase 3 |
| ---------- | ------- | ------- | ------- |
| ResNet-50  | WIP     | WIP     | WIP     | 
| ResNet-101 | WIP     | WIP     | WIP     |
| Swin-Base  | WIP     | WIP     | WIP     |


## Training the Model

We employed 3 steps in the training process:

1. Pretrain the instance segmentation pipeline (COCO)

    *You can get the model in this step by training [DiffusionInst](https://github.com/chenhaoxing/DiffusionInst).*

2. Pretrain the odvis with pseudo key-reference pairs (COCO)
    ```
    python train_net.py --num-gpus 8 --config-file configs/phase_02/coco_r50.yaml
    ```

3. Finetune the model (YTVIS or OVIS)
    ```
    python train_net.py --num-gpus 8 --config-file configs/phase_03/ytvis19_r50.yaml
    ```



## Evaluating the Model

```
python train_net.py --num-gpus 8 --config-file configs/phase_03/ytvis19_r50.yaml --eval-only
```


## Demo

```
python demo.py --config-file configs/phase_02/coco_r50.yaml --input <input.jpg> --output <result.jpg>
```