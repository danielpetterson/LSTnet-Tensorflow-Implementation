
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, GRU, Dropout, Concatenate
import argparse

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
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
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
###-----------------------------------------------------------------------------------
### Layers to transform data for skipGRU
###-----------------------------------------------------------------------------------

class PreSkipTrans(tf.keras.layers.Layer):
    def __init__(self, pt, skip, **kwargs):
        #
        # pt:   Number of different RNN cells = (window / skip)
        # skip: Number of points to skip
        #
        self.pt   = int(pt)
        self.skip = int(skip)
        super(PreSkipTrans, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(PreSkipTrans, self).build(input_shape)
    
    def call(self, inputs):
        # Get input tensors; in this case it's just one tensor
        x = inputs

        # Get the batchsize which is tf.shape(x)[0] since inputs is either X or C which has the same
        # batchsize as the input to the model
        batchsize = tf.shape(x)[0]
        

        # Get the shape of the input data
        input_shape = K.int_shape(x)
        
        # Create output data by taking a 'window' size from the end of input (:-self.pt * self.skip)
        output = x[:,-self.pt * self.skip:,:]
        
        # Reshape the output tensor by:
        # - Changing first dimension (batchsize) from None to the current batchsize
        # - Splitting second dimension into 2 dimensions
        output = tf.reshape(output, [batchsize, self.pt, self.skip, input_shape[2]])


        # Permutate axis 1 and axis 2
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        
        # Reshape by merging axis 0 and 1 now hence changing the batch size
        # to be equal to current batchsize * skip.
        # This way the time dimension will only contain 'pt' entries which are
        # just values that were originally 'skip' apart from each other => hence skip RNN ready
        output = tf.reshape(output, [batchsize * self.skip, self.pt, input_shape[2]])
        
        # Adjust the output shape by setting back the batch size dimension to None
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])
        
        

        return output
    
    def compute_output_shape(self, input_shape):
        # Since the batch size is None and dimension on axis=2 has not changed,
        # all we need to do is set shape[1] = pt in order to compute the output shape
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.pt
        
        return tf.TensorShape(shape)

class PostSkipTrans(tf.keras.layers.Layer):
    def __init__(self, skip, **kwargs):
        #
        # skip: Number of points to skip
        #
        self.skip = skip
        super(PostSkipTrans, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(PostSkipTrans, self).build(input_shape)
    
    def call(self, inputs):
        # Get input tensors
	# - First one is the output of the SkipRNN layer which we will operate on
	# - The second is the oiriginal model input tensor which we will use to get
	#   the original batchsize
        x, batchsize = inputs

        # Get the shape of the input data
        input_shape = K.int_shape(x)

        # Split the batch size into the original batch size before PreTrans and 'Skip'
        output = tf.reshape(x, [batchsize, self.skip, input_shape[1]])
        
        # Merge the 'skip' with axis=1
        output = tf.reshape(output, [batchsize, self.skip * input_shape[1]])
        
        # Adjust the output shape by setting back the batch size dimension to None
        output_shape = tf.TensorShape([None]).concatenate(output.get_shape()[1:])

        return output
    
    def compute_output_shape(self, input_shape):
        # Adjust shape[1] to be equal to shape[1] * skip in order for the 
        # shape to reflect the transformation that was done
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = self.skip * shape[1]
        
        return tf.TransformShape(shape)

###---------------------------------------------------------------------------
class PreCNNReshape(tf.keras.layers.Layer):
    def __init__(self):
        super(PreCNNReshape, self).__init__()
      
    def build(self, input_shape):
        super(PreCNNReshape, self).build(input_shape)
      
    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        result = tf.reshape(inputs, [-1,input_shape[1],input_shape[2],1])
        return result

class PostCNNReshape(tf.keras.layers.Layer):
    def __init__(self):
        super(PostCNNReshape, self).__init__()
      
    def build(self, input_shape):
        super(PostCNNReshape, self).build(input_shape)
      
    def call(self, inputs):
        result = tf.squeeze(inputs,axis=2)
        return result


class Model(tf.keras.Model):

  def __init__(self, args, in_shape):
    super(Model, self).__init__()
    self.m = in_shape[2]
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
    self.reshape1 = PreCNNReshape()
    self.conv1 = Conv2D(self.hidCNN, kernel_size = (self.CNN_kernel, self.m))
    self.relu1 = tf.keras.layers.ReLU()
    self.dropout1 = Dropout(rate = args.dropout)
    self.reshape2 = PostCNNReshape()

    self.gru1 = GRU(self.hidRNN, activation="relu", return_sequences = False, return_state = True)
    self.dropout2 = Dropout(rate = args.dropout)

    if (self.skip > 0):
        self.preskiptrans = PreSkipTrans(self.pt, int((self.window - self.CNN_kernel + 1) / self.pt))
        self.GRUskip = GRU(self.hidSkip, activation="relu", return_sequences = False, return_state = True)
        self.linear1 = Dense(self.m)
        self.postskiptrans = PostSkipTrans(int((self.window - self.CNN_kernel + 1) / self.pt))
    else:
        self.linear1 = Dense(self.m)

    if (self.hw > 0):
        self.highway1 = Flatten()
        self.highway2 = Dense(1)
    # self.output = None
    # if (args.output_fun == 'sigmoid'):
    #     self.output = K.sigmoid
    # if (args.output_fun == 'tanh'):
    #     self.output = K.tanh
    self.concat = Concatenate(axis=1)
    self.flatten = Flatten()
    self.dense_final = Dense(self.m)

  def call(self, inputs):
    # CNN layer
    init = self.reshape1(inputs)
    conv = self.conv1(init)
    conv = self.relu1(conv)
    conv = self.dropout1(conv)
    conv = self.reshape2(conv)

    # GRU layer with Relu activation function
    _,gru = self.gru1(conv)
    gru = self.dropout2(gru)

    # SkipGRU layer with Relu activation function
    if self.skip > 0:
        skipgru = self.preskiptrans(conv)
        _,skipgru = self.GRUskip(skipgru)
        skipgru = self.postskiptrans([skipgru,self.batchsize])

    # Concatenate the outputs of GRU and SkipGRU
    r = self.concat([gru,skipgru])

    # Dense layer
    res = self.flatten(r)
    res = self.dense_final(res)

    if (self.hw > 0):
            z = inputs[:, -int(self.hw):, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z

    return res

### Data
class data_utility(object):
    def __init__(self, filename, train_prop, val_prop, horizon, window):
        self.initdata = np.loadtxt(open(filename),delimiter=',')
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

    

if __name__ == "__main__":
    ### Traffic data
    ### A collection of 48 months (2015-2016) hourly data from the California Department
    ### of Transportation. The data describes the road occupancy rates (between 0 and 1) 
    ### measured by different sensors on San Francisco Bay area freeways.
    window = 24*7
    horizon = 24
    fileloc = "/Users/danielpetterson/GitHub/LSTnet-Tensorflow-Implementation/data/traffic/traffic.txt.gz"
    data = data_utility(fileloc, 0.6 ,0.2,horizon,window)

    model = Model(args, data.train_set[0].shape)


    model.compile(optimizer=args.optim, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    # model_history = []
    # Fit model
    history = model.fit(data.train_set[0], data.train_set[1], epochs=args.epochs, batch_size=args.batch_size, validation_data=(data.val_set[0], data.val_set[1]),verbose=2)
    # model_history.append(history)
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # Plot loss of training and validation sets
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Baseline Configuration')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training','Validation'])
    plt.show()





