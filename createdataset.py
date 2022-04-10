from xml.etree.ElementTree import XML
from preprocess import preprocess
import argparse
import os 
from random import choice
import cv2
from tqdm import trange

XML_FORMAT = """
<annotation>
    <filename>{file_name}</filename>
    <folder>{database}</folder>
    <source>
        <database>{database}</database>
        <annotation>custom</annotation>
        <image>custom</image>
    </source>
    <size>
        <width>{size}</width>
        <height>{size}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    {items}
</annotation>
"""

ITEM_XML_FORMAT = """
    <object>
        <name>{name}</name>
        <pose>unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
"""

# create parser 
parser = argparse.ArgumentParser()

parser.add_argument("--root_dir", help = "Directory where folder is or is to be created", default = './data')
parser.add_argument("--item_folder", help = "folder that has the items, see README for structure information", default = './items')
parser.add_argument("--backgrounds", help = "folder that has the background images, see README for structure information", default = './backgrounds')
parser.add_argument("--dataset_size", help = "number of images to generate in the dataset", default = 10)

args = parser.parse_args()

def create_dataset(root_dir, item_folder, backgrounds, dataset_size = 10, image_size = 640):
    """ creates a dataset from item images and background images
    root_dir: root dir where the ds is created
    item_folfer
    """

    # create needed dir in root_dir 
    for dir in [root_dir, os.path.join(root_dir, 'Annotations'), os.path.join(root_dir, 'JPEGImages'), os.path.join(root_dir, 'ImageSets'), os.path.join(root_dir, 'ImageSets', 'Main')]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # define item names and backgrounds
    backgrounds = list(os.scandir(backgrounds))
    item_names = list(os.scandir(item_folder))
    
    # create labels.txt file \
    with open(os.path.join(root_dir, 'labels.txt'), 'w') as f:
        f.writelines([item.name + '\n' for item in item_names])
    
    for i in trange(dataset_size):

        # randomly select background
        background = choice(backgrounds)

        # randomly select instance of item from each folder
        items = [choice(list(os.scandir(folder))) for folder in item_names]
        
        # preprocces and save image
        photo, labels = preprocess(items, background, output_size = (image_size, image_size))

        photo_file = f'{i:05}.jpeg'
        photo_path = os.path.join(root_dir, 'JPEGImages', photo_file)

        cv2.imwrite(photo_path, photo)

        label_file = f'{i:05}.xml'
        label_path = os.path.join(root_dir, 'Annotations', label_file)

        # create item contents
        item_contents = []

        # loop through each label to create the contents of the xml
        for item_name, label in zip(item_names, labels):
            xmin, ymin, xmax, ymax = label 

            name = str(item_name).replace("<DirEntry ", "").replace(">", "").replace("'", "")
            
            item_xml = ITEM_XML_FORMAT.format(name = name, xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)

            item_contents.append(item_xml)

        item_contents = '\n'.join(item_contents)

        contents = XML_FORMAT.format(file_name = photo_file, database = root_dir, size = image_size, items = item_contents)

        with open(label_path, 'w') as file:
            file.write(contents)

        # write for to splits, this is a hack
        for fname in ['test.txt', 'train.txt', 'trainval.txt', 'val.txt']:
            with open(os.path.join(root_dir, 'ImageSets', 'Main', fname), 'a') as f:
                f.write(f'{i:05}' + '\n')


if __name__ == '__main__':
    create_dataset(args.root_dir, item_folder = args.item_folder, backgrounds = args.backgrounds, dataset_size = int(args.dataset_size))