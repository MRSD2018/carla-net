from PIL import Image
import os
import re
import sys
import os.path
import numpy as np
import multiprocessing
import pdb

data_dir = sys.argv[1]
global output_dir
output_dir = sys.argv[2]
rgb_files = [f for f in os.listdir(data_dir) if re.match(r'carla_rgb', f)]

colors = np.asarray([[  0,   0,   0],
         [128,  64, 128],
         [244,  35, 232],
         [ 70,  70,  70],
         [102, 102, 156],
         [190, 153, 153],
         [153, 153, 153],
         [250, 170,  30],
         [220, 220,   0],
         [107, 142,  35],
         [152, 251, 152],
         [  0, 130, 180],
         [220,  20,  60],
         [255,   0,   0],
         [  0,   0, 142],
         [  0,   0,  70],
         [  0,  60, 100],
         [  0,  80, 100],
         [  0,   0, 230],
         [119,  11,  32],
         [119,  130,  130]])

def checkOrCreate(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


checkOrCreate(output_dir)
checkOrCreate(output_dir + '/carlascapes')
checkOrCreate(output_dir + '/carlascapes/gtFine_trainvaltest')
checkOrCreate(output_dir + '/carlascapes/gtFine_trainvaltest/gtFine')
checkOrCreate(output_dir + '/carlascapes/gtFine_trainvaltest/gtFine/test')
checkOrCreate(output_dir + '/carlascapes/gtFine_trainvaltest/gtFine/train')
checkOrCreate(output_dir + '/carlascapes/gtFine_trainvaltest/gtFine/val')

checkOrCreate(output_dir + '/carlascapes/gtFine_trainvaltest/gtFine/train/carla')

checkOrCreate(output_dir + '/carlascapes/leftImg8bit')
checkOrCreate(output_dir + '/carlascapes/leftImg8bit/test')
checkOrCreate(output_dir + '/carlascapes/leftImg8bit/train')
checkOrCreate(output_dir + '/carlascapes/leftImg8bit/train/carla')
checkOrCreate(output_dir + '/carlascapes/leftImg8bit/val')

global rgb_file_dir
rgb_file_dir = output_dir + '/carlascapes/leftImg8bit/train/carla' 
global gt_file_dir
gt_file_dir = output_dir + '/carlascapes/gtFine_trainvaltest/gtFine/train/carla'

label_mapping = {
  0: 0,
  1: 0,
  3: 0,
  4: 0,
  10: 0,
  13: 0,
  14: 0,
  15: 0,
  16: 0,
  17: 0,
  18: 0,
  19: 0,
  20: 0, 
  7: 7,
  8: 8,
  1: 11,
  11: 12,
  2: 13,
  5: 17,
  12: 20,
  9: 21,
  6: 34
}    

def mapLabels(label):
  return label_mapping[label]

visual_label_mapping = {
  0: colors[3,0:],
  3: colors[3,0:],
  4: colors[3,0:],
  10: colors[3,0:],
  13: colors[3,0:],
  14: colors[3,0:],
  15: colors[3,0:],
  16: colors[3,0:],
  17: colors[3,0:],
  18: colors[3,0:],
  19: colors[3,0:],
  20: colors[3,0:],
  7: colors[1,0:],
  8: colors[2,0:],
  1: colors[3,0:],
  11: colors[4,0:],
  2: colors[5,0:],
  5: colors[6,0:],
  12: colors[8,0:],
  9: colors[9,0:],
  6: colors[20,0:] 
}

 
def mapVisualLabels(label):
  return visual_label_mapping[label];

def processFile(file_name):  


  im = Image.open(data_dir + '/' + file_name)
  num = file_name.split('_')[2].split('.')[0]

  print(num + "/" + str(len(rgb_files)))
  num_pad = str(num).zfill(6) 

  file_root_name = "carla_" + num_pad + "_000020"

  im_file_save = rgb_file_dir + "/" + file_root_name + "_leftImg8bit.png"
  seg_file_save = gt_file_dir + "/" + file_root_name + "_gtFine_labelIds.png"
  seg_visual_file_save = gt_file_dir + "/" + file_root_name + "_gtFine_color.png"


  seg_name = data_dir + '/carla_seg_' + num + '.png'
  if (os.path.isfile(seg_name) and not(os.path.isfile(im_file_save)) and not (os.path.isfile(seg_file_save)) and not (os.path.isfile(seg_visual_file_save))):
    #only save out to dataset if segmentation exists
    seg = Image.open(seg_name)
    seg_arr = np.array(seg)
         
    im.save(im_file_save,"PNG")
    (sizey, sizex, chan) = seg_arr.shape
    seg_arr_labels = np.zeros(seg_arr.shape) 
    seg_arr_human = np.zeros(seg_arr.shape) 
    for j in range(sizey):
      for i in range(sizex): 
        seg_arr_human[j,i,0:] = mapVisualLabels(seg_arr[j,i,0])
        for c in range(chan):
          seg_arr_labels[j,i,c] = mapLabels(seg_arr[j,i,c])
    seg_image = Image.fromarray(seg_arr_labels.astype(np.uint8), mode='RGB')
    seg_image.save(seg_file_save, "PNG")    

    seg_visual_image = Image.fromarray(seg_arr_human.astype(np.uint8), mode='RGB')
    seg_visual_image.save(seg_visual_file_save, "PNG")    

pool = multiprocessing.Pool(8)

pool.map(processFile, rgb_files)
#parse through the files
#for file_name in rgb_files:
#  processFile(file_name, rgb_file_dir, gt_file_dir)
