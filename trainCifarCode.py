__author__ = 'Raj'

from scipy import misc
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf

#import os
#import random
#import matplotlib as mp

# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    #initial = tf.contrib.layers.xavier_initializer(uniform=True,seed=None, dtype=tf.float32)
    W = tf.Variable(initial,dtype=tf.float32)

    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    return h_max

def variable_summaries(var, name):
    """
    Attach summaries to Tensor
    :param var: variable that is in interest
    :param name: name of the variable that one is interested in. must be passed as a string.
    """
    # within the scope of summaries
    with tf.name_scope('summaries'):

        mean = tf.reduce_mean(var)  # Obtain mean of the given variable
        tf.scalar_summary('mean/' + name, mean) # get summary of mean
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))   # Find std of the given variable
        tf.scalar_summary('stddev/' + name, stddev) # get summary of std dev
        tf.scalar_summary('max/' + name, tf.reduce_max(var)) # find max
        tf.scalar_summary('min/' + name, tf.reduce_min(var)) # find min
        tf.histogram_summary(name, var) # find histogram


ntrain = 1000 # per class
ntest = 100 # per class
nclass = 10 # number of classes
imsize = 28
nchannels = 1
batchsize = 100 # maybe 50. But there are 100 images per class in test...?

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

start_time = time.time() # start timing

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = '/Users/rajdbz/Desktop/Coursework/ELEC677/Assignment2/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = '/Users/rajdbz/Desktop/Coursework/ELEC677/Assignment2/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, 1]) #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, shape=[None, 10]) #tf variable for labels

# --------------------------------------------------
# model
#create your model

'''
First Layer
'''
# Conv Layer 5 by 5 and 32 filter maps
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(tf_data, [-1,28,28,1])

# ReLu activation
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Max pooling by 2
h_pool1 = max_pool_2x2(h_conv1)

'''
Second Layer
'''
# Conv Layer 5 by 5 and 64 filter maps
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# ReLu activation
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Max pooling by 2
h_pool2 = max_pool_2x2(h_conv2)

'''
1st Fully connected Layer with ReLu
'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
2nd Fully connected Layer. For training, dropout applied, otherwise not.
'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

keep_prob = tf.placeholder(tf.float32)

h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

h_fc2 = tf.matmul(h_fc1,W_fc2) + b_fc2

'''
Softmax Regression + Softmax Nonlinearity
'''
output = tf.nn.softmax(h_fc2)

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(output), reduction_indices=[1]))

# step size
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(tf_labels,1))

# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add a scalar summary for the snapshot of accuracy and loss
accuracy_summary = tf.scalar_summary(tags='Accuracy', values=accuracy)
loss_summary = tf.scalar_summary(tags='Loss', values=cross_entropy)

# Add statistics summary for the activations
actfun1_summary = variable_summaries(var=h_conv1,name='Activation1')
actfun2_summary = variable_summaries(var=h_conv2,name='Activation2')

# Build the summary operation based on the TF collection of Summaries.
summary_merged = tf.merge_all_summaries()

# --------------------------------------------------
# optimization

# Initialize all the variables
init = tf.initialize_all_variables()

sess.run(init)

# setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_xs = np.zeros([batchsize, imsize, imsize, nchannels])
batch_ys = np.zeros([batchsize, nclass])  # setup as [batchsize, the how many classes]

# define n_iter and n_samples
n_iter = 200
n_samples = 10000  # number of training data set

result_dir = '/Users/rajdbz/Desktop/Coursework/ELEC677/Assignment2/result_cifar10'

# Instantiate a SummaryWriter to output summaries and the Graph.
train_summary_writer = tf.train.SummaryWriter(result_dir + '/training', sess.graph)
test_summary_writer = tf.train.SummaryWriter(result_dir + '/test')

for i in range(n_iter): # try a small iteration size once it works then continue
    perm = np.arange(n_samples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]

    if i%100 == 0:
        # calculate train accuracy and print it
        train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        test_accuracy = accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})

        print("step %d, training accuracy %g" % (i, train_accuracy))
        print("step %d, test accuracy %g" % (i,test_accuracy))

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        '''
        Training summary
        '''
        # obtain full summary (i.e. accuracy and loss)
        train_str = sess.run(summary_merged, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob:0.5})
        train_summary_writer.add_summary(train_str, i)
        #filter_summary = tf.image_summary('filts',W_conv1)
        #train_summary_writer.add_summary(filter_summary)
        train_summary_writer.flush()
        '''
        Test summary
        '''
        test_str = sess.run(summary_merged, feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
        test_summary_writer.add_summary(test_str, i)
        test_summary_writer.flush()

    # dropout only during training
    train_step.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})

# --------------------------------------------------
# test

stop_time = time.time()
print('The training takes %f second to finish' % (stop_time - start_time))

print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))





'''
Obtain the weight images from Convolutional layer 1.
'''

fig = plt.figure()
fig.set_figheight(22)
fig.set_figwidth(10)
fig.tight_layout

for ind in range(32):
    #WTemp2 = tf.get_variable('W_conv1')
    #WTemp = tf.to_int32(W_conv1[:, :, 0, ind])
    WTemp = sess.run(W_conv1)
    #tf.Print(WTemp)
    weight_plot = plt.subplot(8, 4, ind + 1)
    weight_plot.set_title("Conv1 weight %d" % (ind + 1))
    plt.imshow(WTemp[:, :, 0, ind])
    #plt.imshow(WTemp)


    # sess.close()