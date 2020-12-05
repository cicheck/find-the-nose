import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, Input


def build_cascade_model(input_size, r_rate=0.0, dr_rate=0.0):

    # In original model input had size 39
    org_input = 39
    assert input_size in {org_input, org_input * 2, org_input * 4}, ("Input needs to bo one of numbers: "
            + str(org_input) + ", "
            + str(org_input * 2) + ", "
            + str(org_input * 4))
    
    model = Sequential()
    shape = (input_size, input_size, 3)
    model.add(Input(shape=shape))
    # Add pooling layers to follow original architecture
    while model.output_shape[1] > org_input:
        model.add(MaxPool2D(pool_size=(2, 2)))
    assert model.output_shape[1] == org_input, ("Incorrect input shape, "
        + "after initial pooling layers")
    # In original model authors used 3 disjointed networks one for each RGB value.
    # We will use joined version
    multiply_sizes = 3
    # Max pool arguments
    pool_size = (2, 2)
    reg = tf.keras.regularizers.l2(r_rate)
    model.add(Conv2D(kernel_size=(4, 4), activation='relu', filters=20 * multiply_sizes, kernel_regularizer=reg))
    model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
    model.add(Dropout(dr_rate))
    model.add(Conv2D(kernel_size=(3, 3), activation='relu', filters=40 * multiply_sizes, kernel_regularizer=reg))
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(Dropout(dr_rate))
    model.add(Conv2D(kernel_size=(3, 3), activation='relu', filters=60 * multiply_sizes, kernel_regularizer=reg))
    model.add(MaxPool2D(pool_size=pool_size))
    model.add(Dropout(dr_rate))
    model.add(Conv2D(kernel_size=(2, 2), activation='relu', filters=80 * multiply_sizes, kernel_regularizer=reg))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    return model

