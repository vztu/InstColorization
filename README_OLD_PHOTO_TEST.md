## Getting Started
1. Clone this repo:
```sh
git clone https://github.com/vztu/InstColorization.git
cd InstColorization
git checkout old_photo_test
```
2. Install [conda](https://www.anaconda.com/).
3. Install all the dependencies
```sh
conda env create --file env.yml
```
4. Switch to the conda environment
```sh
conda activate instacolorization
```
5. Install other dependencies
```sh
sh scripts/install.sh
```

## Dataset Prepararation

### COCOStuff (ignored)

1. Download and unzip the COCOStuff training set:
```sh
sh scripts/prepare_cocostuff.sh
```
2. Now the COCOStuff trainset would place in train_data.

### Your own dataset

1. Copy `old_photo/data/DIV2K_train_HR.zip` to `./train_data` then unzip.
2. If you want to train on your dataset, you should change the dataset path in scripts/prepare_train_box.sh's L1 and in scripts/train.sh's L1. (Already changed)

## Pretrained Model
1. Download from google drive
```sh
sh scripts/download_model.sh
```
2. Now the pretrained models would place in checkpoints.

## Instance Prediction
Please follow the command below to predict all the bounding boxes fo the images in `${DATASET_DIR}` folder.
```sh
sh scripts/prepare_train_box.sh
```
All the prediction results would save in `${DATASET_DIR}_bbox` folder.