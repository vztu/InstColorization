## Getting Started
1. Clone this repo:
```sh
git clone https://github.com/vztu/InstColorization.git
cd InstColorization
git checkout old_photo_test
```
2. Install [conda](https://www.anaconda.com/).
3. ~~Install all the dependencies~~
```sh
conda env create --file env.yml
```
3. Use our old photo env:
```sh
conda activate old_photo
```

4. Switch to the conda environment
```diff
- conda activate instacolorization
+ conda activate old_photo
```
5. Install other dependencies (modify the detectron2 version refering [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#install-pre-built-detectron2-linux-only), here I have changed to CUDA=11.0 and torch=1.7)
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

1. Copy `old_photo/data/DIV2K_train_HR.zip` to `./train_data` then `unzip`.
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

## Training
Simply run the following command
```sh
sh scripts/train.sh
```
To view training results and loss plots, run `visdom -port 8098` and click the URL \url{http://localhost:8098}.

This is a 3 stage training process.

1. We would start to train our full image colorization branch based on the [siggraph_retrained's pretrained weight](https://github.com/richzhang/colorization-pytorch).
2. We would use the full image colorization branch's weight as our instance colorization branch's pretrained weight.
3. Finally, we would train the fusion module.

## Testing the Instance-aware Image Colorization model
1. Our model's weight would place in `checkpoints/${DATASET_DIR}_mask`.
2. Change the checkpoint's path in [test_fusion.py's L38](test_fusion.py#L38) from `coco_finetuned_mask_256_ffs` to `${DATASET_DIR}_mask`
3. Please follow the command below to colorize all the images in `test_folder` foler based on the weight placed in `${DATASET_DIR}_mask`.

    ```
python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir train_data/DIV2K_resize_val_HR  --results_img_dir results/DIV2K_resize_val_HR
    ```
    All the colorized results would save in `results` folder.