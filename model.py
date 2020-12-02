from keras.models import Input, Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Concatenate
from copy import deepcopy

def conv_33(inputs, ch, acti):
    outputs = Conv2D(ch, 3, activation=acti, padding='same')(inputs)
    outputs = BatchNormalization()(outputs)
    return outputs

def conv_block(inputs, ch, acti):
    outputs = conv_33(inputs, ch, acti)
    outputs = conv_33(outputs, ch, acti)
    return outputs

def build_level(inputs, ch, depth, inc_rate, acti):
    if depth == 0:
        outputs = conv_block(inputs, ch, acti)
    else:
        # down
        concatenate_part = conv_block(inputs, ch, acti)
        outputs = MaxPooling2D()(concatenate_part)

        #recursive
        outputs = build_level(outputs, int(inc_rate * ch), depth - 1, inc_rate, acti)

        #up
        outputs = Conv2DTranspose(ch, 3, strides=2, activation=acti, padding='same')(outputs)
        outputs = Concatenate()([concatenate_part, outputs])
        outputs = conv_block(outputs, ch, acti)
    return outputs

def UNet(img_size = (512, 512, 1), start_conv_ch = 64, out_ch = 1, inc_rate = 2., acti = 'relu', max_depth = 4):
    inputs = Input(img_size)
    outputs = build_level(inputs, start_conv_ch, max_depth, inc_rate, acti)
    outputs = Conv2D(out_ch, 1, activation='sigmoid')(outputs)
    return Model(inputs, outputs)