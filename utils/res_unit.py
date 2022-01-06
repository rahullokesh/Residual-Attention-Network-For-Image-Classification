"""
Version : 1
Version Date : 30th Nov 2021
Description : Utils file for Residual Unit
"""
from tensorflow.keras.layers import Add, Conv2D
from tensorflow.keras.layers import Activation, BatchNormalization

def residual_unit(input_data, filters, conv_stride=1):
    """
    Function for adding Residual Unit before using activation in the residual attention network architecture
    Args:
        input_data: Input to the residual unit
        filters: filter used for the residual unit
        conv_stride: number of strides in the convolution layer

    Returns:
        output: Output obtained from the residual unit which can be passed as an input to the attention module in a given stage
    """
    # Storing the input_data in a separate variable to use it later
    input_identity = input_data

    # doing batch normalization of the input data & passing through the relu activation.
    norm_var_1 = BatchNormalization()(input_data)
    activation_unit_1 = Activation('relu')(norm_var_1)
    # Using 1x1 convolution for channel wise pooling or feature map pooling. Keeping the 'same' padding everywhere
    convolution_1 = Conv2D(filters=filters[0], kernel_size=(1, 1), padding='same')(activation_unit_1)

    # Adding batch normalisation for faster computation/reducing training time. Adding 'relu' non-linearity
    norm_var_2 = BatchNormalization()(convolution_1)
    activation_2 = Activation('relu')(norm_var_2)

    # Changing the strides & the kernel size in accordance with the residual unit type.
    # Higher strides & larger keernel size will reduce computationa; complexity, time & variance in the model
    convolution_2 = Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(conv_stride, conv_stride), padding='same')(activation_2)

    # Adding batch normalisation for faster computation/reducing training time. Adding 'relu' non-linearity
    batch_norm_3 = BatchNormalization()(convolution_2)
    activation_3 = Activation('relu')(batch_norm_3)
    conv_3 = Conv2D(filters=filters[2], kernel_size=(1, 1), padding='same')(activation_3)

    # Changing the dimensions for passing in the input_identity in accordance to the residual unit type
    if input_identity.shape != conv_3.shape:
        filter_conv = conv_3.shape[-1]
        # According to the residual unit type, changing the strides and the kernel size
        input_identity = Conv2D(filters=filter_conv, kernel_size=(1, 1), strides=(conv_stride, conv_stride), padding='same')(input_identity)

    output = Add()([input_identity, conv_3])
    return output
