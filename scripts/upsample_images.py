from PIL import Image
import os
import re
import sys
import os.path
import numpy as np
import multiprocessing
import pdb
from decimal import Decimal
from scipy.misc import imresize 

data_dir = sys.argv[1]
global output_dir
output_dir = sys.argv[2]
rgb_files = [f for f in os.listdir(data_dir) if re.match(r'carla_rgb', f)]

def checkOrCreate(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


checkOrCreate(output_dir)

def processFile(file_name):  
  im = Image.open(data_dir + '/' + file_name)

  im_file_save = output_dir + '/' + file_name
  if not(os.path.isfile(im_file_save)):
    print("Resizing " + file_name)
    im_arr = np.array(im)
    im_out = imresize(im_arr,(1025,2049,3), interp='bilinear')
    im_converted = Image.fromarray(im_out.astype(np.uint8), mode='RGB')
    im_converted.save(im_file_save)
    print("Saved file " + im_file_save)
  
pool = multiprocessing.Pool(8)

pool.map(processFile, rgb_files)
#parse through the files
for file_name in rgb_files:
  processFile(file_name)
