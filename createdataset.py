from xml.etree.ElementTree import XML
import xml.etree.ElementTree as ET

from preprocess import preprocess
import argparse
import os 
from random import choice
import cv2
from tqdm import trange
import shutil

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

parser.add_argument("--root-dir", help = "Directory where folder is or is to be created", default = './data')
parser.add_argument("--item-folder", help = "folder that has the items, see README for structure information", default = './items')
parser.add_argument("--backgrounds", help = "folder that has the background images, see README for structure information", default = './backgrounds')
parser.add_argument("--dataset-size", help = "number of images to generate in the dataset", default = 10)

args = parser.parse_args()

def create_dataset(root_dir, item_folder, backgrounds, dataset_size = 10, image_size = 640):
    """ creates a dataset from item images and background images
    root_dir: root dir where the dataset is to be created
    item_folfer
    """

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # create needed dir in root_dir 
    for dir in [os.path.join(root_dir, 'Annotations'), os.path.join(root_dir, 'JPEGImages'), os.path.join(root_dir, 'ImageSets'), os.path.join(root_dir, 'ImageSets', 'Main')]:
        # remove 
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        
    # define item names and backgrounds
    backgrounds = list(os.scandir(backgrounds))
    item_folders = list(os.scandir(item_folder))
    item_names = [str(item_name).replace("<DirEntry ", "").replace(">", "").replace("'", "") for item_name in item_folders]
    
    # create labels.txt file \
    with open(os.path.join(root_dir, 'labels.txt'), 'w') as f:
        f.write('\n'.join(item_names))
    
    # list if image / label file names 
    file_names = []

    for i in trange(dataset_size):
        # randomly select background
        background = choice(backgrounds)

        # randomly select instance of item from each folder
        items = [choice(list(os.scandir(folder))) for folder in item_folders]
        
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
            
            item_xml = ITEM_XML_FORMAT.format(name = item_name, xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax)

            item_contents.append(item_xml)

        item_contents = '\n'.join(item_contents)

        contents = XML_FORMAT.format(file_name = photo_file, database = root_dir, size = image_size, items = item_contents)

        with open(label_path, 'w') as file:
            file.write(contents)

        file_names.append(f'{i:05}')

    from sklearn.model_selection import train_test_split
    
    # split test and val
    train_names, val_names = train_test_split(file_names, test_size = .1)

    # write split files 
    # test and val are the same
    for fname in ['test.txt', 'val.txt']:
        with open(os.path.join(root_dir, 'ImageSets', 'Main', fname), 'a') as f:
            for name in val_names:
                f.write(name + '\n')

    # write tain names to train file
    with open(os.path.join(root_dir, 'ImageSets', 'Main', 'train.txt'), 'a') as f:
        for name in train_names:
            f.write(name + '\n')

    # trainval will have every file 
    with open(os.path.join(root_dir, 'ImageSets', 'Main', 'trainval.txt'), 'a') as f:
        for name in file_names:
            f.write(name + '\n')

    dataset_info = {
        'num_images': dataset_size,
        'num_classes': len(item_names),
        'classes': item_names,
        'root_dir': root_dir,
    }

    return dataset_info

def read_vox_xml(xml_file: str) -> list:
    """
    xml_file: string in xml structure or path to file in xml structure
    returns: list of lists, sub list is a list of bounding boxes: [xmin, ymin, xmax, ymax]
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []

    for boxes_xml in root.iter('object'):
        # iter over each object in the xml and extract bounding box

        ymin = int(boxes_xml.find("bndbox/ymin").text)
        xmin = int(boxes_xml.find("bndbox/xmin").text)
        ymax = int(boxes_xml.find("bndbox/ymax").text)
        xmax = int(boxes_xml.find("bndbox/xmax").text)

        box = [xmin, ymin, xmax, ymax]
        boxes.append(box)

    return boxes # list 


def test_bounding_box(image_file, bounding_box_path):
    """"""
    image = cv2.imread(image_file)
    
    # parse bounding boxes 
    bounding_boxes = read_vox_xml(bounding_box_path)

    # iterate through bounding boxes, draw bounding box on image
    for xmin, ymin, xmax, ymax in bounding_boxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    # display image
    cv2.imshow('bounding box', image)
    
    cv2.waitKey(1000)






if __name__ == '__main__':
    
    #create_dataset(args.root_dir, item_folder = args.item_folder, backgrounds = args.backgrounds, dataset_size = int(args.dataset_size))

    """
    python createdataset.py --root-dir /mnt/c/Users/14135/Desktop/pytorch-ssd/data --dataset-size 100
    """

    for i in range(1000000):
        test_bounding_box(image_file = f'data/JPEGImages/{i:05}.jpeg', bounding_box_path = f'data/Annotations/{i:05}.xml')
        input()

        