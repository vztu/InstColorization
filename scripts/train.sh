DATASET_DIR=train_data/DIV2K_resize_train_HR
DATASET_NAME="div2k_resize_train_HR"
BATCH_SIZE=4

# # Stage 1: Training Full Image Colorization 
# # [NOTE]: if train from scratch: do not use arg "--load_model"
mkdir ./checkpoints/${DATASET_NAME}_full
# # cp ./checkpoints/siggraph_retrained/latest_net_G.pth ./checkpoints/${DATASET_NAME}_full/
# # python train.py --stage full --name ${DATASET_NAME}_full --sample_p 1.0 --niter 100 --niter_decay 50 --load_model --lr 0.0005 --model train --fineSize 256 --batch_size ${BATCH_SIZE} --display_ncols 3 --display_freq 1600 --print_freq 1600 --train_img_dir $DATASET_DIR
python train.py --stage full --name ${DATASET_NAME}_full --sample_p 1.0 --niter 100 --niter_decay 50 --lr 0.0005 --model train --fineSize 256 --batch_size ${BATCH_SIZE} --display_ncols 3 --display_freq 1600 --print_freq 1600 --train_img_dir $DATASET_DIR


# Stage 2: Training Instance Image Colorization
mkdir ./checkpoints/${DATASET_NAME}_instance
cp ./checkpoints/${DATASET_NAME}_full/latest_net_G.pth ./checkpoints/${DATASET_NAME}_instance/
python train.py --stage instance --name ${DATASET_NAME}_instance --sample_p 1.0 --niter 100 --niter_decay 50 --load_model --lr 0.0005 --model train --fineSize 256 --batch_size ${BATCH_SIZE} --display_ncols 3 --display_freq 1600 --print_freq 1600 --train_img_dir $DATASET_DIR

# # # Stage 3: Training Fusion Module
# mkdir ./checkpoints/${DATASET_NAME}_mask
# cp ./checkpoints/${DATASET_NAME}_full/latest_net_G.pth ./checkpoints/${DATASET_NAME}_mask/latest_net_GF.pth
# cp ./checkpoints/${DATASET_NAME}_instance/latest_net_G.pth ./checkpoints/${DATASET_NAME}_mask/latest_net_G.pth
# cp ./checkpoints/${DATASET_NAME}_full/latest_net_G.pth ./checkpoints/${DATASET_NAME}_mask/latest_net_GComp.pth
# python train.py --stage fusion --name ${DATASET_NAME}_mask --sample_p 1.0 --niter 10 --niter_decay 20 --lr 0.00005 --model train --load_model --display_ncols 4 --fineSize 256 --batch_size 1 --display_freq 500 --print_freq 500 --train_img_dir $DATASET_DIR