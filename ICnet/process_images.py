import re
import sys, os
import os.path
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

def checkOrCreate(directory): 
  if not os.path.exists(directory): 
    os.makedirs(directory) 


try:
    import pydensecrf.densecrf as dcrf
except:
    print("Failed to import pydensecrf,\
           CRF post-processing will not work")

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))

    rgb_files = [f for f in os.listdir(args.img_path) if re.match(r'carla_rgb', f)]


    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes

    # Setup Model
    model = get_model(model_name, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()

    checkOrCreate(args.out_path)
    for file_name in rgb_files:
      im_file_save = args.out_path + '/' + file_name
      if not(os.path.isfile(im_file_save)):
        print("Loading image " + file_name)
  
        img = misc.imread(args.img_path + '/' + file_name)
        resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp='bicubic')
        orig_size = img.shape[:-1]
    
        if model_name in ['pspnet', 'icnet', 'icnetBN']:
            img = misc.imresize(img, (orig_size[0]//2*2+1, orig_size[1]//2*2+1)) # uint8 with RGB mode, resize width and height which are odd numbers
        else:
            img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= loader.mean
        if args.img_norm:
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()
    
        if torch.cuda.is_available():
            model.cuda(0)
            images = Variable(img.cuda(0), volatile=True)
        else:
            images = Variable(img, volatile=True)
    
        outputs = model(images)
        #outputs = F.softmax(outputs, dim=1)
    
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        if model_name in ['pspnet', 'icnet', 'icnetBN']:
            pred = pred.astype(np.float32)
            pred = misc.imresize(pred, orig_size, 'nearest', mode='F') # float32 with F mode, resize back to orig_size
        decoded = loader.decode_segmap(pred)
        print('Classes found: ', np.unique(pred))
        misc.imsave(im_file_save, decoded)
        print("Segmentation Mask Saved at: {}".format(im_file_save))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')



    # Architecture
    parser.add_argument('--arch', nargs='?', type=str, default='icnet',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')

    # Datasets
    parser.add_argument('--dataset', nargs='?', type=str, default='carlascapes',
                        help='Dataset to use [\'carlascapes, cityscapes, pascal, camvid, ade20k etc\']')

    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--dcrf', dest='dcrf', action='store_true', 
                        help='Enable DenseCRF based post-processing | False by default')
    parser.add_argument('--no-dcrf', dest='dcrf', action='store_false', 
                        help='Disable DenseCRF based post-processing | False by default')
    parser.set_defaults(dcrf=False)

    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    args = parser.parse_args()
    test(args)
