import cv2
import os
import numpy as np
import random

def center_crop(img):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = img.shape[0] if img.shape[0]<img.shape[1] else img.shape[1]
	mid_x, mid_y = int(width / 2), int(height / 2)
	cw2, ch2 = int(crop_width/2), int(crop_width / 2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def hue_shift(image: cv2, shift: float) -> cv2:
    """
     image: numpy file representing an image (cv2 file)
     shift: float range (0 to 1)
    :return: image with shifted hue
    """

    # convert to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # shift hue
    h_new = cv2.add(h, shift * 180)

    # combine new hue with s and v
    hsv_new = cv2.merge([h_new, s, v])

    # convert from HSV to BGR
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

def brightness_shift(image: cv2, brightness: float) -> cv2:
    """
    Args:
     image: numpy file representing an image (cv2 file)
     brightness: beta (0-1)
    :return: image with shifted brightness
    """
    return cv2.convertScaleAbs(image, beta=brightness*100)

def contrast_shift(image: cv2, contrast: float) -> cv2:
    """
     image: numpy file representing an image (cv2 file)
     contrast: alpha (0-1)
    :return: image with shifted contrast
    """
    return cv2.convertScaleAbs(image, alpha=contrast*2+1)

def rotate(image, theta):
    angle = theta *180/np.pi
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def shift(image, y_shift_factor, x_shift_factor):
    y_shift = int(y_shift_factor * image.shape[1])
    x_shift = int(x_shift_factor * image.shape[0]) 
    translation_matrix = np.float32([ [1,0,y_shift], [0,1, x_shift]])
    return cv2.warpAffine(image, translation_matrix, image.shape[:2])

def noise(image, sigma = 3, mean = 0):
    gaussian = np.clip(image + np.random.normal(mean, sigma, size = image.shape), 0, 255)
    gaussian = np.rint(gaussian)
    gaussian = gaussian.astype(np.uint8)
    return gaussian

def remove_randbox(item, height, width):
    """
    Args: 
        item: 4 channel cv2 image of item
        height: height of box to subtract
        width: width of box to subtract
    """

    # define shape of item 
    shape = item.shape

    # to locations for the center of the object 
    center_x, center_y = np.random.rand(2)

    # scale values to pixels 
    center_x *= shape[1]
    center_y *= shape[0]
    width *= shape[1]
    height *= shape[0]
   
    # cast values to ints 
    low_x, high_x = int(center_x - width/2) , int(center_x + width/2)
    low_y, high_y = int(center_y - height/2), int(center_y + height/2)

    # account for edge conditions
    low_x, high_x, low_y, high_y = [max(x, 0) for x in [low_x, high_x, low_y, high_y]]

    # create mask 
    mask = np.ones(shape, dtype= type(item[0][0][0]))

    # redefine shape for edge condigtons 
    shape = mask[low_y: high_y, low_x: high_x, :].shape

    # generate random pixel values for over lay and reshape to correct shape
    random_noise = np.random.randint(0, 255, shape, dtype = type(item[0][0][0]))
    random_noise = random_noise.reshape(shape)

    # set box to random noise
    item[low_y: high_y, low_x: high_x, :] = random_noise

    # if the transparent layer exists, write it to 1 (full occupancy)
    if item.shape[2] == 4:
        item[low_y: high_y, low_x: high_x, 3] = np.ones(item[low_y: high_y, low_x: high_x, 3].shape)

    return item

def overlay(background, img, scale, center_x = .5, center_y = .5):
    """ 
    args:
        background: background to overlay onto shape [ydim, xdim, 3]
        img: overlay image shape [:, :, 4] where fourth layer is transparency mask 
        scale: relative size of img compared to background in range(0:1), img is resized 
        center_x, center_y : relative position to place img in background in range(0:1)
    returns: 
        background overlayed with mask, shape [:, :, 3]
    """
    assert img.shape[-1] == 4, "img has incorrect labels"

    h, w = [int(background.shape[i] * scale) for i in range(2)]

    y = int(background.shape[0] * center_y - h/2)
    x = int(background.shape[1] * center_x - w/2)

    img = cv2.resize(img, (w, h))

    # make transparency mask binary
    mask = img[:, :, 3:] // 255 

    # repeating the mask over channels
    mask = np.repeat(mask, 3, axis = -1) 

    # mask for the background,  ones and zeros 
    mask_inv = (mask - 1) * -1 

    # apply mask to overlay
    ov = img[:, :, :3] * mask 

    # variables to account for edge conditions
    x_l, y_l = max(0, -x), max(0, -y) # if x > 0, no boundry issue, else limit edges
    x_u, y_u = max(background.shape[1] - x - w, w), max(background.shape[0] - y - h, h)

    x_u = background.shape[1] - x - w if background.shape[1] < x + w else w
    y_u = background.shape[0] - y - h if background.shape[0] < y + h else h

     # prevent negative index on background
    x_, y_ = max(0, x), max(0, y)

    # apply overlay
    background[y_:y+h, x_:x+w, :] = ov[y_l:y_u, x_l:x_u, :3] + background[y_:y+h, x_:x+w, :] * mask_inv[y_l:y_u, x_l:x_u, :3]

    return background


def augment(image, rotate_range = np.pi/12, 
                        brightness_range = .2,
                        hue_range = 0.05, 
                        contrast_range = .2,
                        shift_range = 0,
                        random_flip = False,
                        add_noise = True):
    image = image # prevent hazards

    image = rotate(image, random.uniform(-rotate_range, rotate_range))


    # apply random shift 
    x_shift = random.uniform(-shift_range, shift_range)
    y_shift = random.uniform(-shift_range, shift_range)
    image = shift(image, x_shift, y_shift)

    # save copy of image for proccess later
    image_orignal = image.copy()

    image = brightness_shift(image, random.uniform(-brightness_range, brightness_range)) # random adjust background brightness
    image = hue_shift(image, random.uniform(-hue_range, hue_range)) # random adjust background hue 
    image = contrast_shift(image, random.uniform(-contrast_range, contrast_range))
    if add_noise: image = noise(image, random.uniform(0, 10))

    if random_flip:
        if random.random() < .5:
            image = cv2.flip(image, 0)
        if random.random() < .5:
            image = cv2.flip(image, 1)
    
    # add back orignal alpha layer
    if image_orignal.shape[-1] == 4:
        image_orignal[:, :, :3] = image[:, :, :3]
        image = image_orignal

    return image

def preprocess(item_files, 
            background_file, 
            output_size = (640, 640),
            scale_range = (.1, .3),
            root_dir = './'):
    """
    Args:
        item_files: filenames of object images 
        background_file: path to image of background 
        output_size: tuple: (pixels, pixels) path to image of output_size,
        scale_range: tuple of floats: range to scale image by [0, 1)
            change scale range to reflect realistic item sizes 
        """

    # read the back 
    background_path = os.path.join(root_dir, background_file)
    background = cv2.imread(background_path)

    # center crop then sizesize background
    background = center_crop(background) 
    background = cv2.resize(background, output_size)

    # define background size 
    background_shape = background.shape

    # apply transforms to background 
    background = augment(background)

    # create list to store labels in
    labels = []

    # iterate over item paths 
    for item_file in item_files:
        # read item 
        item_path = os.path.join(root_dir, item_file)
        item = cv2.imread(item_path, -1)

        # apply custom augmentations
        item = augment(item, rotate_range = np.pi)

        # remove random box to simulate partial occlusion
        item = remove_randbox(item, random.uniform(.3, .7), random.uniform(.3, .7))   

        # choice x and y randomly
        x_center, y_center = [random.uniform(0., 1.) for i in range(2)]

        # random choise relative size, could be improed by having smaller values be smaller 
        scale = random.uniform(*scale_range)
        
        # overlay item onto background 
        background = overlay(background, item, scale, x_center, y_center)

        # calculate xmin, ymin, xmax, ymax
        xmin, ymin = x_center - scale / 2, y_center - scale / 2
        xmax, ymax = x_center + scale / 2, y_center + scale / 2

        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(1, xmax), min(1, ymax)

        xmin, ymin, xmax, ymax = [int(var * background_shape[0]) for var in (xmin, ymin, xmax, ymax)]

        # append label to labels list
        labels.append((xmin, ymin, xmax, ymax))

    return background, labels

if __name__ == '__main__':
    x = 1
