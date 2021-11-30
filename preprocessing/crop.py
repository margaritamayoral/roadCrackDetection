"""
This file provides a utility function which crops the images according to the
agreed dimensions
"""

import os
import time

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

source_path = "/strNet/home/dlopez/data/raw_grey_data"
destination_path = "/strNet/home/dlopez/data/grey_512"

extensions = ["png", "jpg"]
destination_extension = 'png'

resize_factor = 8

def crop_images(images, resize_factor=resize_factor, extensions=extensions,
        destination_extension=destination_extension,
        source_path=source_path, destination_path=destination_path):
    start_time = time.time()
    count = 0
    tot = 4 * len(images)

    for image_filename in images:
        if os.path.exists(image_filename):
            image = Image.open(image_filename)
            width, height = image.size
            for i in range(4):
                cropped = image.crop((i * (width // 4), height // 4,
                                      (i+1) * (width // 4), 3 * (height // 4)))
                resized = cropped.resize((width // 4 // resize_factor,
                                          height // 2 // resize_factor))
                resized.save(
                    os.path.join(destination_path, 'image{}.{}'.format(
                        str(count).zfill(5), destination_extension)))

                count += 1
                if count % 10 == 0:
                    print('We are {:.4f}% done, it has taken {:2f}s'.format(
                        100 * count / tot, time.time() - start_time))


if __name__ == "__main__":
    images = []

    for f in os.listdir(source_path):
        if any(f.endswith(extension) for extension in extensions):
            images.append(os.path.join(source_path, f))

    for folder in ["images/", "masks/"]:
        full_src_path = os.path.join(source_path, folder)
        images = []
        for f in os.listdir(full_src_path):
            images.append(os.path.join(full_src_path, f))
        crop_images(images, source_path=full_src_path,
                destination_path=os.path.join(destination_path, folder))

    crop_images(images)
