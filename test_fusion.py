import os
from os.path import join
import time
from options.train_options import TrainOptions, TestOptions
from models import create_model
from util.visualizer import Visualizer
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from tqdm import trange, tqdm

from fusion_dataset import Fusion_Testing_Dataset
from util import util
from utils import ssim, loss, color_space_convert
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', True)

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    opt = TestOptions().parse()
    save_img_path = opt.results_img_dir
    if os.path.isdir(save_img_path) is False:
        print('Create path: {0}'.format(save_img_path))
        os.makedirs(save_img_path)
    opt.batch_size = 1

    # extract bbox
    cmd_bbox = f"python inference_bbox.py --test_img_dir {opt.test_img_dir}"
    os.system(cmd_bbox)

    dataset = Fusion_Testing_Dataset(opt)
    
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    dataset_size = len(dataset)
    print('#Testing images = %d' % dataset_size)

    model = create_model(opt)
    # model.setup_to_test('coco_finetuned_mask_256')
    model.setup_to_test('pascal_clean_more_val_train_scratch_mask')

    # use SSIM and PSNR to evluate image
    ssim_metric = ssim.SSIM().cuda()
    lpips_metric = loss.lipis_eval('alex').cuda()
    hsv = color_space_convert.RgbToHsv(training=False)

    count_empty = 0
    time_test = 0
    ssim_acc_rgb = 0.
    psnr_acc_rgb = 0.
    ssim_acc_gray = 0.
    psnr_acc_gray = 0.
    lpips_acc = 0.
    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        # if os.path.isfile(join(save_img_path, data_raw['file_id'][0] + '.png')) is True:
        #     continue
        with torch.no_grad():
            start_time = time.time()
            data_raw['full_img'][0] = data_raw['full_img'][0].cuda()
            if data_raw['empty_box'][0] == 0:
                data_raw['cropped_img'][0] = data_raw['cropped_img'][0].cuda()
                box_info = data_raw['box_info'][0]
                box_info_2x = data_raw['box_info_2x'][0]
                box_info_4x = data_raw['box_info_4x'][0]
                box_info_8x = data_raw['box_info_8x'][0]
                cropped_data = util.get_colorization_data(data_raw['cropped_img'], opt, ab_thresh=0, p=opt.sample_p)
                full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
                model.set_input(cropped_data)
                model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                model.forward()
            else:
                count_empty += 1
                full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
                model.set_forward_without_box(full_img_data)
            time_test += time.time() - start_time
            
            model.save_current_imgs(join(save_img_path, data_raw['file_id'][0] + '.png'))

    # split two loops to avoid OOM
    for data_raw in tqdm(dataset_loader, dynamic_ncols=True):
        with torch.no_grad():
            # copy from colab ex. Resize to meet original image size
            img = cv2.imread(join(opt.test_img_dir, data_raw['file_id'][0]+'.png'))
            gt_image = TF.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze_(0).cuda()
            lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, _, _ = cv2.split(lab_image)

            img = cv2.imread(join(save_img_path, data_raw['file_id'][0]+'.png'))
            lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            _, a_pred, b_pred = cv2.split(lab_image)
            a_pred = cv2.resize(a_pred, (l_channel.shape[1], l_channel.shape[0]))
            b_pred = cv2.resize(b_pred, (l_channel.shape[1], l_channel.shape[0]))
            gray_color = np.ones_like(a_pred) * 128

            gray_image = cv2.cvtColor(np.stack([l_channel, gray_color, gray_color], 2), cv2.COLOR_LAB2BGR)
            color_image = cv2.cvtColor(np.stack([l_channel, a_pred, b_pred], 2), cv2.COLOR_LAB2BGR)
            rgb_image = TF.to_tensor(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)).unsqueeze_(0).cuda()
            cv2.imwrite(join(save_img_path, data_raw['file_id'][0] + '.png'), color_image)

            ssim_acc_rgb += ssim_metric(rgb_image, gt_image)
            ssim_acc_gray += ssim_metric(hsv(rgb_image), hsv(gt_image))

            psnr_acc_rgb += loss.batch_psnr(rgb_image, gt_image, 1.)
            psnr_acc_gray += loss.batch_psnr(hsv(rgb_image), hsv(gt_image), 1.)

            lpips_acc += lpips_metric(rgb_image, gt_image)

    print('{0} images without bounding boxes'.format(count_empty))
    with open(join(save_img_path, 'log.txt'), 'w') as f:
        print('The average RGB SSIM is %.3f,  RGB PSNR is %.3f, '
          'The average GRAY SSIM is %.3f,  GRAY PSNR is %.3f,'
          'lpips is %.3f and average inference time is %.3f'
          % (ssim_acc_rgb / dataset_size, psnr_acc_rgb / dataset_size,
             ssim_acc_gray / dataset_size, psnr_acc_gray / dataset_size,
             lpips_acc / dataset_size, time_test / dataset_size),  file=f)
