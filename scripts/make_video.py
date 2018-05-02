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
label_dir = sys.argv[2]
output_dir = sys.argv[3]

def checkOrCreate(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

checkOrCreate(output_dir)

rgb_files = [f for f in os.listdir(data_dir) if re.match(r'carla_rgb', f)]
  



def processFile(file_name):  
  print("Processing image " + file_name)
  im = Image.open(data_dir + '/' + file_name)
  label = Image.open(label_dir + '/' + file_name)
  im_arr = np.array(im)
  label_arr = np.array(label)
  (sizey, sizex, chan) = im_arr.shape
  for j in range(sizey):
    for i in range(sizex):
      #print(label_arr[j,i,:])
      if label_arr[j,i,0] == 124 and label_arr[j,i,1] == 0 and label_arr[j,i,2] == 0: 
        im_arr[j,i,:] = [0,0,255]
  im_out = imresize(im_arr,(512,1024,3), interp='bilinear')
  im_converted = Image.fromarray(im_out.astype(np.uint8), mode='RGB')
  num = Decimal(file_name.split('_')[2].split('.')[0])
  num_pad = str(num).zfill(6)
  im_converted.save(output_dir + '/' + num_pad + '.png')
  print('Saved file ' + num_pad + '.png')


pool = multiprocessing.Pool(8)

pool.map(processFile, rgb_files)
#for file_name in rgb_files:
#  processFile(file_name)


