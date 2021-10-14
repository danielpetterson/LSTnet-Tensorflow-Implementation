
# Import libraries
# Import libraries
import numpy as np
import pandas as pd
import argparse
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputLayer, Conv1D, Conv2D, Dense, Flatten, GRU, Dropout, Concatenate, Add
from keras.models import load_model

# Set seeds
tf.random.set_seed(123)
random.seed(123)
np.random.seed(123)


### Default arguments from original Pytorch implementation--------------------------------------
parser = argparse.ArgumentParser(description='Tensorflow Time series forecasting')
parser.add_argument('--data', type=str,
                    help='location of the data file')
parser.add_argument('--model', type=str, default='LSTNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=100,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=24 * 7,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=6,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()

class PreCNNReshape(tf.keras.layers.Layer):
  def __init__(self, batchsize):
    super(PreCNNReshape, self).__init__()
    self.batchsize=batchsize
    
  def build(self, input_shape):
    super(PreCNNReshape, self).build(input_shape)
    
  def call(self, inputs):
    inputshape = K.int_shape(inputs)
    result = tf.reshape(inputs, [-1,inputshape[1],inputshape[2],1])
    return result

class PostCNNReshape(tf.keras.layers.Layer):
  def __init__(self):
    super(PostCNNReshape, self).__init__()
    
  def build(self, input_shape):
    super(PostCNNReshape, self).build(input_shape)
    
  def call(self, inputs):
    result = tf.squeeze(inputs,axis=2)
    return result

class PreSkipGRUReshape(tf.keras.layers.Layer):
  def __init__(self, pt, skip):
    super(PreSkipGRUReshape, self).__init__()
    self.pt   = int(pt)
    self.skip = int(skip)
      
  def build(self, input_shape):
    super(PreSkipGRUReshape, self).build(input_shape)
  
  def call(self, inputs):
    batchsize = tf.shape(inputs)[0]
    inputshape = K.int_shape(inputs)
    out = inputs[:,-self.pt * self.skip:,:]
    out = tf.reshape(out, [batchsize, self.pt, self.skip, inputshape[2]])
    out = tf.transpose(out, perm=[0,2,1,3])
    result = tf.reshape(out, [batchsize * self.skip, self.pt, inputshape[2]])
    return result

class PostSkipGRUReshape(tf.keras.layers.Layer):
  def __init__(self, skip):
    super(PostSkipGRUReshape, self).__init__()
    self.skip = skip

  def build(self, input_shape):
    super(PostSkipGRUReshape, self).build(input_shape)

  def call(self, inputs):
    input = inputs
    inputshape = K.int_shape(input)
    out = tf.reshape(input, [-1, self.skip, inputshape[1]])
    result = tf.reshape(out, [-1, self.skip * inputshape[1]])
    return result

class PreAutoRegReshape(tf.keras.layers.Layer):
  def __init__(self, highway):
    super(PreAutoRegReshape, self).__init__()
    self.hw = int(highway)
    
  def build(self, input_shape):
    super(PreAutoRegReshape, self).build(input_shape)
    
  def call(self, inputs):
    inputs, inputshape = inputs
    out = inputs[:,-self.hw:,:]
    out = tf.transpose(out,perm=[0,2,1])
    result = tf.reshape(out,[inputshape[0]*K.int_shape(inputs)[2],self.hw])
    return result

class PostAutoRegReshape(tf.keras.layers.Layer):
  def __init__(self, m):
    super(PostAutoRegReshape, self).__init__()
    self.m = m
    
  def build(self, input_shape):
    super(PostAutoRegReshape, self).build(input_shape)
    
  def call(self, inputs):
    inputs, inputshape = inputs
    result = tf.reshape(inputs, [inputshape[0], self.m])
    return result

class Model(tf.keras.Model):

  def __init__(self, args, in_shape, dilation):
    super(Model, self).__init__()
    self.m = in_shape[2]
    self.dilation = dilation
    self.batchsize = args.batch_size
    self.window = args.window
    self.hidRNN = args.hidRNN
    self.hidCNN = args.hidCNN
    self.hidSkip = args.hidSkip
    self.CNN_kernel = args.CNN_kernel
    self.skip = args.skip
    self.pt = (self.window - self.CNN_kernel)/self.skip
    self.hw = args.highway_window
    self.inputshape = in_shape
    self.input1 = InputLayer(input_shape=in_shape[1:], batch_size=self.batchsize)
    self.reshape1 = PreCNNReshape(self.batchsize)
    self.dilconv1 = Conv2D(self.hidCNN, kernel_size = (int(self.CNN_kernel), int(self.m//2)), dilation_rate=dilation, padding="same", activation="relu")
    self.conv1 = Conv2D(self.hidCNN, kernel_size = (self.CNN_kernel, self.m), activation="relu")
    self.reshape2 = PostCNNReshape()
    self.reshape3 = PreSkipGRUReshape(self.pt, int((self.window - self.CNN_kernel + 1) / self.pt))
    self.dropout1 = Dropout(rate = args.dropout)
    self.gru1 = GRU(self.hidRNN, activation="relu", return_sequences = False, return_state = True)
    self.dropout2 = Dropout(rate = args.dropout)
    if (self.skip > 0):
        self.GRUskip = GRU(self.hidSkip, activation="relu", return_sequences = False, return_state = True)
        self.linear1 = Dense(self.m)
        self.reshape4 = PostSkipGRUReshape(int((self.window - self.CNN_kernel + 1) / self.pt))
    else:
        self.linear1 = Dense(self.m)
    if (self.hw > 0):
        self.reshape5 = PreAutoRegReshape(self.hw)
        self.highway1 = Flatten()
        self.highway2 = Dense(1)
        self.reshape6 = PostAutoRegReshape(self.m)
    self.concat = Concatenate(axis=1)
    self.flatten = Flatten()
    self.dense_final = Dense(self.m)
    self.sum = Add()

  def call(self, inputs):

    # Input layer
    first = self.input1(inputs)

    # CNN layer
    init = self.reshape1(first)
    if self.dilation >= 2:
      conv = self.dilconv1(init)
      conv = self.conv1(conv)
    else:
      conv = self.conv1(init)
    conv = self.dropout1(conv)
    conv = self.reshape2(conv)

    # GRU layer with Relu activation function
    _,gru = self.gru1(conv)
    gru = self.dropout2(gru)

    # SkipGRU layer with Relu activation function
    if self.skip > 0:
      skipgru = self.reshape3(conv)
      _,skipgru = self.GRUskip(skipgru)
      skipgru = self.reshape4(skipgru)

    # Concatenate the outputs of GRU and SkipGRU
    r = self.concat([gru,skipgru])

    # Dense layer
    res = self.flatten(r)
    res = self.dense_final(res)
    
    # Highway
    if (self.hw > 0):
        z = self.reshape5([inputs, tf.shape(inputs)])
        z = self.highway1(z)
        z = self.highway2(z)
        z = self.reshape6([z,tf.shape(inputs)])
          
    res = self.sum([res,z])

    return res

### Data
class data_utility(object):
    def __init__(self, filename, train_prop, val_prop, horizon, window, header=False):
        if header == True:
            self.initdata = pd.read_csv(filename, skiprows=1, delimiter=',').values
        else:
            self.initdata = pd.read_csv(filename, skiprows=0, delimiter=',').values
        self.data = np.zeros(self.initdata.shape)
        self.n, self.m = self.initdata.shape
        self.scale = np.ones(self.m)
        for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.initdata[:,i]))
                self.data[:,i] = self.initdata[:,i] / np.max(np.abs(self.initdata[:,i]))
        self.horizon = horizon
        self.window = window
        self.split_data(train_prop, val_prop)

    def split_data(self, train_prop, val_prop):
        train_set = range(self.window + self.horizon - 1, int(train_prop * self.n))
        val_set = range(int(train_prop * self.n), int((train_prop + val_prop) * self.n))
        test_set  = range(int((train_prop + val_prop) * self.n), self.n)
        set_lst = []
        for i in [train_set, val_set, test_set]:
            x = np.zeros((len(i), self.window, self.m))
            y = np.zeros((len(i), self.m))
            for j in range(len(i)):
                end   = i[j] - self.horizon + 1
                start = end - self.window
                
                x[j,:,:] = self.data[start:end, :]
                y[j,:]   = self.data[i[j],:]
            set_lst.append([x, y])
        self.train_set = set_lst[0]
        self.val_set = set_lst[1]
        self.test_set = set_lst[2]

### Metrics
# Source: https://github.com/gucoloradoc/AQForecasting/blob/18d4ef8084adcdbf36903d13bdc6515aec13b00b/Monterrey/ANN_output/27-06-2019_22-41-43/new_generator.py
def RSE(y_true, y_pred):
    RSE = K.sqrt(K.sum(K.square(y_pred-y_true)))
    return RSE

# Source: https://github.com/WenYanger/Keras_Metrics/blob/master/PearsonCorr.py
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)
    

if __name__ == "__main__":
    ### Traffic data
    ### A collection of 24 months (2015-2016) hourly data from the California Department
    ### of Transportation. The data describes the road occupancy rates (between 0 and 1) 
    ### measured by different sensors on San Francisco Bay area freeways. This dataset uses
    ### data from the first 25 sensors of the original dataset that can be found here:
    ### 

    path = "https://danielpetterson.github.io/assets/traffic.csv"

    def eval(data_path, args, dilation, horizon, save_loc):
        data = data_utility(data_path, 0.6 ,0.2,horizon=horizon,window=24*7, header=True)
        # Instantiate
        model = Model(args, data.train_set[0].shape, dilation=dilation)
        # Compile
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                        loss=tf.keras.losses.MeanAbsoluteError(), metrics=[RSE, pearson_r])
        
        checkpoint_filepath = save_loc
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            mode='max',
            save_best_only=True)
        
        # Fit model
        history = model.fit(data.train_set[0], data.train_set[1],
                            epochs=args.epochs, batch_size=args.batch_size, 
                            validation_data=(data.val_set[0], data.val_set[1]),
                            callbacks=[model_checkpoint_callback])
        
        # Save architecture and weights
        model.save(save_loc)
        score = model.evaluate(data.test_set[0], data.test_set[1], verbose = 0) 
        print(score)

        return history, score

    hist2_12, score2_1 = eval(path, args, 2, 12, '/content/gdrive/MyDrive/model12')
    hist_12, score1 = eval(path, args, 1, 12, '/content/gdrive/MyDrive/lst12')
    hist2_24, score2_2 = eval(path, args, 2, 24, '/content/gdrive/MyDrive/model24')
    hist_24, score1_2 = eval(path, args, 1, 24, '/content/gdrive/MyDrive/lst24')

    # # Plot loss of training and validation sets
    # plt.plot(history.history['RSE'])
    # plt.plot(history.history['val_RSE'])
    # plt.title('')
    # plt.ylabel('Root Square Error')
    # plt.xlabel('Epoch')
    # plt.legend(['Training','Validation'])
    # plt.show()





