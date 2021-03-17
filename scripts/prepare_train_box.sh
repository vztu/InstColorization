DATASET_DIR=train_data/DIV2K_train_HR

python inference_bbox.py --test_img_dir $DATASET_DIR --filter_no_obj
# python inference_bbox.py --test_img_dir example
