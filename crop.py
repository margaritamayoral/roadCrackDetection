from PIL import Image
import os
import time

source_path = "/strNet/bulk_dev/MA_Burlington" + \
        "/011/20190515_MA_Burlington_Office_0/0/l360_pano"
destination_path = "/strNet/home/dlopez/data/colour"

extensions = ["png", "jpg"]
destination_extension = 'png'

resize_factor = 8

images = []

for f in os.listdir(source_path):
    if any(f.endswith(extension) for extension in extensions):
        images.append(os.path.join(source_path, f))

start_time = time.time()
count = 0
tot = 4 * len(images)

for image_filename in images:
    image = Image.open(image_filename)#.convert('LA')
    width, height = image.size
    for i in range(4):
        cropped = image.crop((i * (width // 4), height // 4,
                              (i+1) * (width // 4), 3 * (height // 4)))
        resized = cropped.resize((width // 4 // resize_factor,
                                  height // 2 // resize_factor))
        resized.save(os.path.join(destination_path, 'image{}.{}'.format(
            str(count).zfill(5), destination_extension)))

        count += 1
        if count % 10 == 0:
            print('We are {:.4f}% done, it has taken {:2f}s'.format(
                100 * count / tot, time.time() - start_time))
