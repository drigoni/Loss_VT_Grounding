# Loss_VT_Grounding
This repository contains the code used to generate the results reported in the paper: [A Better Loss for Visual-Textual Grounding]() 

# Dependencies
This project uses the `conda` environment.
In the `root` folder you can find the `.yml` file for the configuration of the `conda` environment and also the `.txt` files for the `pip` environment. 
The environment is currently set for CPU only.

# Structure
The project is structured as follows: 
* `backup`: contains the file used to extract the image features from [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) model.
* `data`: contains datasets and pre-processed files;
* `model_code`: contains code about the model;
* `model_code/parser`: contains code about the parsing of the datasets;
* `results`: contains checkpoints and the results.

# Usage
NOTE: in order to execute correctly the code, the users need to set in the code their absolute path to this folder: `Loss_VT_Grounding`.

### Proposal Extraction
In order to extract the bounding boxes proposals, we have used the following implementation: [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).
In particular, we have followed the guide reported in that repository, and we have applied the model to both Flickr30k and Referit images, respectively.
In folder `backup` there are the files used to extract the bounding box proposal features: `BU_extract_feat.py` and `BU_demo.py`.
The commands are:
```
python ./tools/extract_feat.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net ./data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --img_dir ./data/flickr30k/flickr30k_images/ --out_dir ./data/flickr30k/features/ --num_bbox 100,100 --feat_name pool5_flat
python ./tools/extract_feat.py --gpu 0 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net ./data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --img_dir ./data/referit/referitGame/images/ --out_dir ./data/referit/features/ --num_bbox 100,100 --feat_name pool5_flat
```
where the folder `/data/referit/referitGame/images/` contains all the Referit dataset images and `./data/flickr30k/flickr30k_images/` contains the Flickr30k images.
We used their pre-trained model.

### Data Download
First you need to download the necessary datasets. In particular, it is needed Flickr30k entities dataset and Referit dataset, respectively.
The final structure should be:
```
Loss_VT_Grounding
|-- data
    |-- flickr30k
        |-- flickr30k_entities
            |-- Annotations
            |-- Sentences
            |-- test.txt
            |-- train.txt
            |-- val.txt
        |-- flickr30k_images
    |-- flickr30k_raw
        |-- out_bu
        |-- preprocessed
    |-- refer 
    |-- referit_raw
        |-- out_bu
        |-- preprocessed
```
The bottom-up-attention extracted features need to be placed in the following folders:
* `data/flickr30k_raw/out_bu/`: for Flickr30k, and
* `data/referit_raw/out_bu/`: for Referit.

`refer` is the following repository: [https://github.com/lichengunc/refer](https://github.com/lichengunc/refer). 
The user just need to download the images from http://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip and unzip them in folder `data/images/saiapr_tc-12`.

**UPDATE (16/11/2022)**: Setting up the environment for visual feature extraction requires some settings that are not easy for many users. If you can't compile it, you can jump to this repository (I haven't had a chance to test it yet): https://github.com/manekiyong/bottom-up-attention-docker
For the remaining brave ones who prefer to compile the package from scratch, I can only wish you good luck!


### Environment
To configure the environment:
```bash
conda env create -f env.yml 
conda activate vtkel
pip install -r env.txt
```

### Data Pre-processing
In order to generate the final pre-processed data, type the following commands:
```bash
python make_dataset_flickr30k.py
python make_dataset_referit.py
```
The generated files are placed in `flickr30k_raw/preprocessed/` and `referit/preprocessed/`.

### Model Training
In order to train the model use:
```bash
python trainer.py --configs '{"mode":0, "dataset":"flickr"}'
python trainer.py --configs '{"mode":0, "dataset":"referit"}'
```

### Model Test
In order to test the model:
```bash
python trainer.py --configs '{"mode":1, "dataset":"flickr", "restore": "/home/user/repository/Loss_VT_Grounding/results/model_flickr_9.pth"}'
python trainer.py --configs '{"mode":1, "dataset":"referit", "restore": "/home/user/repository/Loss_VT_Grounding/results/model_referit_9.pth"}'
```

### Show Examples
In order to display some test examples:
```bash
python trainer.py --configs '{"mode":2, "dataset":"flickr", "restore": "/home/user/repository/Loss_VT_Grounding/results/model_flickr_9.pth"}'
python trainer.py --configs '{"mode":2, "dataset":"referit", "restore": "/home/user/repository/Loss_VT_Grounding/results/model_referit_9.pth"}'
```

# Pre-processed datasets, Pre-trained Models and Results
Unfortunately, the pre-processed data requires more than 100GB and can't be uploaded with the code.
However, there are all the files needed to reproduce every result obtained (we have fixed the seeds).

To download the pre-trained weights: [https://www.dropbox.com/s/2whyp6jsap3ateo/SAC2022_grounding_pre-trained_models..zip?dl=0](https://www.dropbox.com/s/2whyp6jsap3ateo/SAC2022_grounding_pre-trained_models..zip?dl=0).

# Information
For any questions and comments, contact [Davide Rigoni](mailto:davide.rigoni.2@phd.unipd.it).

# License
MIT
