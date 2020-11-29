import numpy as np
from PIL import Image
import random
from preprocess import resize_picture


def select_image_part(file_name, coordinates, min_size):
    """Capture part of image of size at least min_size * min_size
    
    Returns suc, img, cord
    Where:
    suc -> True if selection was successfull, False if not
    img_arr -> new image ass array if selection was successful
    coord -> new coordinates if selection was successful
    """ 
    img = Image.open(file_name)
    # Covert into RGB
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background
    # Check if is large enough for selection
    if img.size[0] < min_size + 20 or img.size[1] < min_size + 20:
        return False, None, None
    new_image_width = random.randint(min_size, img.size[0] - 10)
    new_image_height = random.randint(min_size, img.size[1] - 10)
    x_shift = random.randint(0, img.size[0] - new_image_width - 1)
    y_shift = random.randint(0, img.size[1] - new_image_height - 1)
    crop_rectangle = (x_shift, y_shift, x_shift + new_image_width, y_shift + new_image_height)
    # Check if face centre lay inside new picture
    if coordinates[0] < x_shift + 5 or coordinates[0] > x_shift + new_image_width - 5:
        return False, None, None
    if coordinates[1] < y_shift + 5 or coordinates[1] > y_shift + new_image_width - 5:
        return False, None, None
    img = img.crop(crop_rectangle)
    img_arr = np.asarray(img)
    return True, img_arr, [coordinates[0] - x_shift, coordinates[1] - y_shift]


def shift_colors(x, shift_range=20):
    """Shift RGB values in array x"""
    
    x = x.copy()
    # Final part ensures that padding will stay in shifted picture
    shift1 = np.random.rand(x.shape[0], x.shape[1]) * shift_range * (x[:, :, 0] > 0.1)
    shift2 = np.random.rand(x.shape[0], x.shape[1]) * shift_range * (x[:, :, 1] > 0.1)
    shift3 = np.random.rand(x.shape[0], x.shape[1]) * shift_range * (x[:, :, 2] > 0.1)
    x[:, :, 0] = x[:, :, 0] + shift1
    x[:, :, 1] = x[:, :, 1] + shift2
    x[:, :, 2] = x[:, :, 2] + shift3
    # Ensure that values are in correct range
    x[x > 255.0] = 255.0
    x[x < 0.0] = 0.0
    return x


def mirror(x, coordinates):
    """Flip horizontally array x
    
    Returns flipped array and point corresponding to
    marked point in original array
    """
    img = Image.fromarray(x.astype(np.uint8))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_arr = np.asarray(img)
    
    return img_arr, [img_arr.shape[0] - coordinates[0], coordinates[1]]


def apply_select_image_part(labels_df, min_size, tries_number=4):
    """For each img try to apply select_image_part tries_number times"""

    new_pictures = []
    new_coordinates = []
    for index, row in labels_df.iterrows():
        for _ in range(tries_number):
            suc, img_arr, coord = select_image_part(labels_df['file_name'][index], [labels_df['x_coord'][index], labels_df['y_coord'][index]], min_size)
            if suc:
                img_arr, x_scale, y_scale = resize_picture(img_arr, min_size)
                new_pictures.append(img_arr)
                new_coordinates.append([coord[0] * x_scale, coord[1] * y_scale])            
        
    S_X = np.zeros((len(new_pictures), min_size, min_size, 3))
    S_Y = np.zeros((len(new_pictures), 2))
    print(len(new_pictures))
    for i in range(len(new_pictures)):
        S_X[i, :] = new_pictures[i]
        S_Y[i, :] = new_coordinates[i]
    return S_X, S_Y