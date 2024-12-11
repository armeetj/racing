import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
print(sys.path)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

def detect_drivable_area(cfg, opt):
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'yolo_test')
    device = select_device(logger, opt.device)

    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)

    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()

    dataset = LoadImages(opt.source, img_size=opt.img_size)
    
    t0 = time.time()
    for path, img, img_det, _, shapes in tqdm(dataset, total=len(dataset)):
        img = transform(img).to(device).unsqueeze(0)

        # Run inference
        t1 = time_synchronized()
        _, da_seg_out, _ = model(img)
        t2 = time_synchronized()

        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        img_det = show_seg_result(img_det, (da_seg_mask, None), _, _, is_demo=True)

        save_path = os.path.join(opt.save_dir, Path(path).name)
        cv2.imwrite(save_path, img_det)

    print(f'Results saved to {opt.save_dir}')
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/End-to-end.pth', help='model weights path')
    parser.add_argument('--source', type=str, default='inference/images', help='source folder for images')
    parser.add_argument('--img-size', type=int, default=640, help='inference image size')
    parser.add_argument('--device', default='cpu', help='device to run inference on')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    opt = parser.parse_args()

    with torch.no_grad():
        detect_drivable_area(cfg, opt)