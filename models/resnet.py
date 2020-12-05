from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50


def build_resnet_model(input_shape):

    backbone = ResNet50(include_top=False, input_shape=input_shape, weights="imagenet")
    
    # Freeze weights
    for layer in backbone.layers:
        if layer.name not in ['conv5_block3_out', 'conv5_block3_add', 'conv5_block3_3_bn', 'conv5_block3_3_conv']:
            layer.trainable = False

    # Head
    head = Sequential()
    head.add(Flatten())
    head.add(Dense(240, activation='relu'))
    head.add(Dense(120, activation='relu'))
    head.add(Dense(2, activation='sigmoid'))

    # Model
    model = Sequential([backbone, head])
    return model