"""
This file creates masks from a json object containing all label information
"""

import os
import json
import csv
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw

WIDTH = 8192
HEIGHT = 4096

NEW_WIDTH = 2048
NEW_HEIGHT = 1024

thickness = 4

image_src_root = '/strNet'

data_root = '/strNet/home/dlopez/data/'
mask_json = 'masks.json'
mask2_json = 'masks2.json'
new_mask_json = 'new_masks.json'
label_names_csv = 'label_names.csv'

raw_images = 'images/'
masks = 'masks/'

extension = 'png'

old_data_root = '/strNet/bulk_dev/Training_Data/'

old_data_folders = [folder_name + 'masks/' for folder_name
        in ['test/', 'test3/', 'test4/']]

with open(os.path.join(data_root, mask_json)) as f:
    labels = json.loads(f.read())

with open(os.path.join(data_root, mask2_json)) as f:
    labels = labels + json.loads(f.read())

# Save label values and names in a csv file for future usage
label_names = {}
for label in labels:
    label_val = label['label_val']
    name_extended = label['label_name']
    label_names[label_val] = name_extended

with open(os.path.join(data_root, label_names_csv), 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['label_val', 'label_name'])
    writer.writeheader()
    for k, v in sorted(label_names.items()):
        writer.writerow({'label_val': k, 'label_name': v})

# Create dict where keys are filenames,
# and values are all the labels in that file
filenames = {}
for label in labels:
    filename = os.path.join(image_src_root,
            label['path2file'].replace('\\', '/'))
    if filename not in filenames.keys():
        filenames[filename] = []
    filenames[filename].append({
        "points": label['bounding_box'],
        "label_val": label['label_val'],
        # "name_extended": label["name_extended"],
        # "category": label["category"],
        # "subcategory": label["subcategory"],
    })

with open(os.path.join(data_root, new_mask_json), 'w') as f:
    json.dump(filenames, f)

count = 0;
for filename in filenames.keys():
    if os.path.exists(filename):
        count += 1

print("{}% recovered, {} out of {}".format(
    100 * count / len(filenames), count, len(filenames)))


print(len(filenames))
# find all paths to old masks
old_mask_filenames = []
for folder in old_data_folders:
    full_dir_path = os.path.join(old_data_root, folder)
    for f in os.listdir(full_dir_path):
        if f.endswith(extension):
            old_mask_filenames.append(os.path.join(full_dir_path, f))
print(len(old_mask_filenames))
count = 0

filename_set = set(filenames.keys())

for old_mask_file in old_mask_filenames:
    raw_array = np.asarray(Image.open(old_mask_file), dtype=np.int32)
    old_array = np.clip(raw_array, a_min=0, a_max=1) * 255
    Image.fromarray(old_array.astype(np.uint8)).save(
            'test_images/true{}.png'.format(count))
    min_dist = 0
    min_img = None
    min_f = None

    print('start creating masks')
    for original_img_file in filename_set:
        data = filenames[original_img_file]
        img = Image.new('L', (WIDTH, HEIGHT), 0)
        for datum in data:
            dr = ImageDraw.Draw(img)
            points = list(map(tuple, datum['points']))
            dr.polygon(points,
                    outline=255, fill=255)
            dr.line(points, fill=255, width=thickness)
        img_resized = img.resize((NEW_WIDTH, NEW_HEIGHT))
        img_array = np.asarray(img_resized, dtype=np.int32)
        intersection = np.sum(
                np.multiply(img_array / 255.0, old_array / 255.0))
        union = np.sum(np.clip(np.add(
            img_array / 255.0, old_array / 255.0), a_min=0, a_max=1))

        if union > 0:
            dist = intersection / union
        else:
            dist = 0
        if dist > min_dist:
            min_dist = dist
            min_f = original_img_file
            min_img = img_resized
    if min_img != None:
        min_img.save('test_images/test{}.png'.format(count))
    print(min_dist, min_f, old_mask_file, count)

    dest_directory = os.path.dirname(min_f)
    src_path = os.path.join(os.path.dirname(os.path.dirname(old_mask_file)),
            'images',
            'img{}.png'.format(re.findall("(\d+)",
                os.path.basename(old_mask_file))[0]))

    if min_dist > 0.5:
        filename_set.remove(min_f)
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        shutil.copyfile(src_path, min_f)

    count += 1


    # img = Image.new('L', (WIDTH, HEIGHT), 0)
    # print('creating masks')
    # for datum in data:
    #     ImageDraw.Draw(img).polygon(list(map(tuple, datum['points'])),
    #             outline=255, fill=255)
    # print('turning into arrays')
    # img_array = np.asarray(img.resize((NEW_WIDTH, NEW_HEIGHT)),
    #         dtype=np.int32)
    # img.resize((NEW_WIDTH, NEW_HEIGHT)).save('test{}.png'.format(count))


    # print(img_array.dtype)
    # min_dist = float('inf')
    # min_f = None
    # for f_ in old_masks_filenames:
    #     old_array = np.clip(np.asarray(Image.open(f_), dtype=np.int32),
    #             a_min=0, a_max=1) * 255
    #     dist = np.sum.square((img_array - old_array)/255.0))
    #     if dist < min_dist:
    #         min_dist = dist
    #         min_f = f_
    # count += 1
    # print(min_dist, min_f, f, count)
    # print('new', img_array.shape)






