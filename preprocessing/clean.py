"""
This python file cleans the dataset
In other words, it makes sure the "no signs" label is not drawn
"""

import os
import json
import csv
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm

from crop import crop_images

WIDTH = 8192
HEIGHT = 4096

NEW_WIDTH = 2048
NEW_HEIGHT = 1024

thickness = 1

extension = 'png'

image_src_root = '/strNet'

data_root = os.path.join(image_src_root, 'home/dlopez/data/')
mask_json = 'masks.json'
mask2_json = 'masks2.json'
new_mask_json = 'new_masks.json'
label_names_csv = 'label_names.csv'

raw_data = os.path.join(data_root, 'raw_grey_data/')
new_data_root = os.path.join(data_root, 'test0/')

raw_images = os.path.join(raw_data, 'images/')
raw_masks = os.path.join(raw_data, 'masks/')

with open(os.path.join(data_root, new_mask_json)) as f:
    filenames = json.loads(f.read())

count = 0;
areas = {}

print('Generating masks and images...')
for filename, data in tqdm(filenames.items()):
    if os.path.exists(filename):
        new_filename = 'img{0:06d}.{1}'.format(count, extension)
        try:
            original_image = Image.open(filename).convert('LA')
        except OSError:
            continue
        original_image.save(os.path.join(raw_images, new_filename))
        img = Image.new('L', (WIDTH, HEIGHT), 0)
        for datum in data:
            test_img = Image.new('L', (WIDTH, HEIGHT), 0)
            test_dr = ImageDraw.Draw(test_img)
            points = list(map(tuple, datum['points']))
            test_dr.polygon(points,
                    outline=255, fill=255)
            test_dr.line(points, fill=255, width=thickness)
            test_array = np.asarray(test_img, dtype=np.int32)
            
            mean = np.mean(test_array / 255)
            if mean < 0.75:
                if mean in areas:
                    areas[mean].append(filename)
                else:
                    areas[mean] = [filename]
            #     dr = ImageDraw.Draw(img)
            #     points = list(map(tuple, datum['points']))
            #     dr.polygon(points,
            #             outline=255, fill=255)
            #     dr.line(points, fill=255, width=thickness)
        #img.save(os.path.join(raw_masks, new_filename))
        
        count += 1
import matplotlib.pyplot as plt

plt.plot(list(sorted(areas.keys())))
plt.savefig('means.png')

print("{}% recovered, {} out of {}".format(
    100 * count / len(filenames), count, len(filenames)))
