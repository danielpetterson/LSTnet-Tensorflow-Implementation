# Import libraries
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2d, Dense, Flatten, GRU, Dropout, Reshape


def Model(input_shape):
    # CNN layer
    init = Input(shape = input_shape[1:])
    conv = Conv2d(activation="relu")(init)
    conv = Dropout()(conv)
    conv = Reshape(K.int_shape(conv)[1], K.int_shape(conv)[3])
    # GRU layer with Relu activation function
    rnn = GRU(activation="relu", return_sequences = False, return_state = True)(conv)
    rnn = Dropout()

    # SkipGRU

