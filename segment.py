"""function that takes in a jpeg and segments out a piece of it
the object will be placed on a colored photo with qr code on the corners"""
import cv2
import numpy as np
import subprocess
import os
import shutil 


def segment(image_path, output_path = "output1.png"):

    ps = subprocess.Popen(('ps', '-A'), stdout=subprocess.PIPE)
    
    # segment image
    cmd = ['backgroundremover',
        '-i', image_path, 
        '-o', output_path]

    cmd_output = subprocess.check_output(cmd, stdin=ps.stdout)

    # read in output image
    image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
 
    # find where the non zeros indexes are in the transparency layer 
    y, x = image[:, :, 3].nonzero()

    # find the min and max pixels that have non zero values
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    
    
    # crop the image removing the transparent space 
    cropped_image = image[ymin:ymax, xmin:xmax]

    # fill to square
    width = xmax - xmin
    height = ymax - ymin 

    xpad = max(height - width, 0) // 2
    ypad = max(width - height, 0) // 2

    cropped_image = cv2.copyMakeBorder(cropped_image, ypad, ypad, xpad, xpad, cv2.BORDER_CONSTANT, None, value = 0)

    # saved cropped image 
    cv2.imwrite(output_path, cropped_image)


def main(image_in_dir, image_out_dir = "./itemstest"):
    # list the items in image_in_dir as strings 
    items = [str(item_name).replace("<DirEntry '", "").replace("'>", "") for item_name in os.scandir(image_in_dir)]

    # clear old objects if exsits 
    dir = image_out_dir
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    # iter through each image on the way in
    for item in items:
        # clear output dir of any contents 
        dir = os.path.join(image_out_dir, item)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        photos_in_path = os.path.join(image_in_dir, item)
        photo_files = [str(file).replace("<DirEntry '", "").replace("'>", "") for file in os.scandir(photos_in_path)]

        # iterate through each photo file 
        for i, photo_in_file in enumerate(photo_files):
            photo_in_path = os.path.join(image_in_dir, item, photo_in_file)
            photo_out_path = os.path.join(image_out_dir, item, f'{i}.png')

            # segment the image and write the item to correct folder
            segment(photo_in_path, photo_out_path)

            print(f"Photo from {photo_in_path} segmented and cropped into {photo_out_path}")

if __name__ == '__main__':
    main(
        image_in_dir = 'items_in',
        image_out_dir = 'items'
        )