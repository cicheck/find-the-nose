import numpy as np
from PIL import Image  

def resize_picture(img_array, new_size):
    """Resize picture into given size
    
    returns transformed picture and scale used during transformation
    """
    img = Image.fromarray(img_array)
    # resize image
    original_size = img.size
    img.thumbnail((new_size, new_size))
    # scale
    scale_x = img.size[0] / original_size[0]
    scale_y = img.size[1] / original_size[1]
    image_arr = np.asarray(img)
    # Add padding
    data = np.zeros((new_size, new_size, 3))
    data[:image_arr.shape[0], :image_arr.shape[1], :] = image_arr
    return data, scale_x, scale_y


def preprocess_picture(file_name, new_size):
    """Transform given picture into new_size * new_size array
    
    returns transformed picture and scale used during transformation
    """
    img = Image.open(file_name)
    # Covert into RGB
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background
    return resize_picture(np.asarray(img), new_size)


def preprocess_df(labels_df, img_size):
    """Preprocess labeled pictures from df and save them as numpy arrays.

    Returns:
        X -- preprocessed pictures
        Y -- scaled labels
    """
    m = len(labels_df)
    X = np.zeros((m, img_size, img_size, 3))
    Y = np.zeros((m, 2))

    for index, row in labels_df.iterrows():
        file_name = row['file_name']
        picture_arr, scale_x, scale_y = preprocess_picture(file_name, img_size)
        x_cord = row['x_coord'] * scale_x
        y_cord = row['y_coord'] * scale_y
        X[index, :] = picture_arr
        Y[index, 0] = x_cord
        Y[index, 1] = y_cord
    return X, Y