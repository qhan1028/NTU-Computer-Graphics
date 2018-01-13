#
#   Fully Convolutional Networks for Semantic Segmentation: Test Data Reader
#   Written by Qhan
#

from __future__ import print_function
import os
import os.path as osp
import numpy as np
import cv2
import PIL.Image as Image

'''
:filename: test data list
:resize_size:
:return: np array of images, names, original size
'''
def read_data(dirpath):
    
    images, names, count = [], [], 0

    for filename in os.listdir(dirpath):
        
        name, ext = osp.splitext(filename)

        if ext not in ['.jpg', '.png']: continue

        print('\r' + ' ' * 50, end='', flush=True)
        print('\rread image: ' + filename, end='', flush=True)
        
        image = cv2.imread(dirpath + '/' + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        names.append(name)
        images.append(image)
        count += 1

    print('\r' + ' ' * 50, end='', flush=True)
    print('\rtotal: %d' % count)

    return np.array(images), np.array(names)
