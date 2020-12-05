import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from .preprocess import preprocess_picture, change_to_rgb
from tensorflow.keras.applications.resnet import preprocess_input


MAX_RGB = 255.0


def plot_landmarks(x, y):
    """Plot face with marked landmarks.
    
    Where y is a list of landmarks coordinates
    and x is numpy array representing a picture
    """
    fig, ax = plt.subplots(sharex=True,sharey=True)
    ax.imshow(x[:,:,:])
    for i in range(len(y)):
        ax.scatter(y[i][0], y[i][1], marker='X', c='r', s=10)


def plot_face(x, coordinates):
    """Plot face with marked center.
    
    Where x is numpy array representing single picture
    """
    fig, ax = plt.subplots(sharex=True,sharey=True)
    ax.imshow(x[:,:,:].astype('uint8'))
    ax.scatter(coordinates[0], coordinates[1], marker='X', c='r', s=10)


def plot_faces(X, Y, rows_number=5, columns_number=5):
    """Plot rows_number * columns_number faces with marked center.
    
    Faces are chosen randomly.
    """
    n = 0
    irand=np.random.choice(Y.shape[0],rows_number*columns_number)
    fig, ax = plt.subplots(rows_number,columns_number,sharex=True,sharey=True,figsize=[rows_number*2,columns_number*2])
    for row in range(rows_number):
        for col in range(columns_number):
            ax[row,col].imshow(X[irand[n],:,:,:].astype('uint8'))
            ax[row,col].scatter(Y[irand[n],0], Y[irand[n],1], marker='X',c='r',s=10)
            ax[row,col].set_xticks(())
            ax[row,col].set_yticks(())
            ax[row,col].set_title('image index = %d' %(irand[n]),fontsize=10)
            n += 1


def plot_prediction(x, y_pred):
    """Plot single face with marked predicted centre.
    
    Where x is numpy array representing a picture
    """
    fig, ax = plt.subplots(sharex=True,sharey=True)
    ax.imshow(x[:,:,:].astype('uint8'))
    ax.scatter(y_pred[0], y_pred[1], marker='X', c='b', s=10)


def plot_predictions(X, Y, Y_pred, rows_number=5, columns_number=5):
    """Plot faces with marked labeled and predicted centre.
    
    Where X is numpy array representing a pictures
    """
    n = 0
    irand=np.random.choice(Y.shape[0],rows_number*columns_number)
    fig, ax = plt.subplots(rows_number,columns_number,sharex=True,sharey=True,figsize=[rows_number*2,columns_number*2])
    for row in range(rows_number):
        for col in range(columns_number):
            ax[row,col].imshow(X[irand[n],:,:,:].astype('uint8'))
            ax[row,col].scatter(Y[irand[n],0], Y[irand[n],1], marker='X',c='r',s=10)
            ax[row,col].scatter(Y_pred[irand[n],0], Y_pred[irand[n],1], marker='X',c='b',s=10)
            ax[row,col].set_xticks(())
            ax[row,col].set_yticks(())
            ax[row,col].set_title('image index = %d' %(irand[n]),fontsize=10)
            n += 1


def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of  metrics vs. epoch
    
    Arguments:
    epochs -- epochs list
    hist -- training history given as pd.DataFrame
    list_of_metics -- list of metrics to plot
    """    
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    # Plot given metrics
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()


def test_model(file_name, model, picture_size):
    """Test model on single picture
    
    Plot face with predicted point
    """
    x, scale_x, scale_y = preprocess_picture(file_name, picture_size)
    x = x / MAX_RGB
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    x, y_pred = x * MAX_RGB, y_pred * picture_size
    # load original image
    img = Image.open(file_name)
    # Covert into RGB
    img = change_to_rgb(img)
    img_arr = np.asarray(img)
    # Re scale
    y_pred[0, 0] = y_pred[0, 0] / scale_x
    y_pred[0, 1] = y_pred[0, 1] / scale_y
    plot_prediction(img_arr, y_pred[0, :])


def test_tl_model(file_name, model, picture_size):
    """Test transfer learning model on single picture

    Plot face with predicted point
    """
    x, scale_x, scale_y = preprocess_picture(file_name, picture_size)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    x, y_pred = x * MAX_RGB, y_pred * picture_size
    # load original image
    img = Image.open(file_name)
    # Covert into RGB
    img = change_to_rgb(img)
    img_arr = np.asarray(img)
    # Re scale
    y_pred[0, 0] = y_pred[0, 0] / scale_x
    y_pred[0, 1] = y_pred[0, 1] / scale_y
    plot_prediction(img_arr, y_pred[0, :])