import os
import time

# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import Tensorflow and start a session
import tensorflow as tf
sess = tf.InteractiveSession()

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

    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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

    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE

    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def NonLinearity(type):

    return{
            'tanh': tf.tanh,
            'sigmoid': tf.sigmoid,
            'relu': tf.nn.relu,
            'elu': tf.nn.elu,
            'softplus': tf.nn.softplus
            }[type]

def optimizer_type(learning_rate, type):

    return{
            'GDO': tf.train.GradientDescentOptimizer(learning_rate),
            'Adam': tf.train.AdamOptimizer(learning_rate),
            'Adagrad': tf.train.AdagradOptimizer(learning_rate)

            }[type]


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

def accuracy_summaries(softmax_output, correct_class, name):
    """
    :param softmax_output: output of the softmax at the end of the pipeline
    :param correct_class: true classes
    :param name: desired name you want.
    :return: accuracy
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(softmax_output, 1), tf.argmax(correct_class, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy/'+name, accuracy)

    return accuracy

def nn_layer(input_tensor, input_dim, layer_name, actFunc=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.

    :param input_tensor = input tensor
    :param input_dim = dimension of weights
    :param layer_name = name of the layer

    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer

        with tf.name_scope('weights'):    # obtain weights and get summary of weights
            weights = weight_variable(input_dim)
            variable_summaries(weights, layer_name + '/weights')

        with tf.name_scope('biases'):     # obtain biases and get summary of biases
            biases = bias_variable([input_dim[3]])
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):  # obtain net inputs and get summary of the inputs
            net_inputs = conv2d(input_tensor, weights) + biases
            # net_inputs = pre_activation = Wx_plus_b
            variable_summaries(net_inputs, layer_name+ '/pre_activations_(net_inputs)')

        with tf.name_scope('activations'): # Activate with input act. This can be changed for now it is relu
            activations = actFunc(net_inputs, name='activations')
            variable_summaries(activations, layer_name+ '/activations')

        with tf.name_scope('max_pool_2x2'): # obtain max pool and get summary the summary of max pool
            max_pool = max_pool_2x2(activations)
            variable_summaries(max_pool, layer_name + '/max pooling')

        return max_pool

def main():
    # Specify training parameters
    result_dir = './results2/' # directory where the results from the training are saved
    max_step = 1100 # the maximum iterations. After max_step iterations, the training will stop no matter what
    learning_rate = 1e-4
    optimizer = 'Adam'
    act = NonLinearity('relu')

    start_time = time.time() # start timing

    # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK

    # placeholders for input data and input labels
    x = tf.placeholder(tf.float32, shape = [None, 784]) # 2D floating tensor. 28 by 28 MNIST image 28*28=784
    y_ = tf.placeholder(tf.float32, shape = [None, 10]) # 10 numbers to be classified. output.

    # reshape the input image
    # Reshape x to a 4d tensor with 2nd and 3rd dim corresponding to image width and height,
    # and final dim corresponding to the number of color channels.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    '''
    monitor the statistics (min, max, mean, standard deviation, histogram) of the following terms after each
    100 iterations: weights, biases, net inputs at each layer , activations after ReLU at each layer, activations
    after Max-Pooling at each layer
    '''

    # first convolutional layer
    conv1_Relu_maxpool=nn_layer(input_tensor=x_image, input_dim=[5,5,1,32],
                                layer_name='conv1_Relu_maxpool', actFunc = act)
    conv2_Relu_maxpool=nn_layer(input_tensor=conv1_Relu_maxpool, input_dim=[5,5,32,64],
                                layer_name='conv2_Relu_maxpool', actFunc = act)


    # densely connected layer: fc(1024)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(conv2_Relu_maxpool, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu( tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

    # setup training

    # obtain mean cross entropy
    diff = y_ * tf.log(y_conv)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(diff, reduction_indices=[1]))

    # summary of cross entropy
    variable_summaries(cross_entropy,'/cross_entropy')

    # Train step size:
    train_step = optimizer_type(learning_rate, optimizer).minimize(cross_entropy)

    accuracy = accuracy_summaries(softmax_output=y_conv, correct_class=y_,name='blahblah')

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(result_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # run the training
    for i in range(max_step):
        batch = mnist.train.next_batch(50) # make the data batch, which is used in the training iteration.
                                            # the batch size is 50
        if i%100 == 0:
            # output the training accuracy every 100 iterations
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_:batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        # save the checkpoints every 1100 iterations
        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

            # output the validation error every epoch

            # test set accuracy
            test_img = mnist.test.images
            test_label = mnist.test.labels
            test_error = 1 - accuracy.eval(feed_dict={x: test_img, y_: test_label, keep_prob: 1.0})
            print("step %d, test error %g" % (i, test_error))
            test_summary = tf.scalar_summary('test error', test_error)
            TEST_error = sess.run(test_summary, feed_dict={x: test_img, y_: test_label, keep_prob: 1.0})
            summary_writer.add_summary(TEST_error, i)

            # validation set accuracy
            val_img = mnist.validation.images
            val_label = mnist.validation.labels
            val_error = 1 - accuracy.eval(feed_dict={x: val_img, y_: val_label, keep_prob: 1.0})
            print("step %d, validation error %g" % (i, val_error))
            val_summary = tf.scalar_summary('validation error', val_error)
            VAL_error = sess.run(val_summary, feed_dict={x: val_img, y_: val_label, keep_prob: 1.0})
            summary_writer.add_summary(VAL_error, i)
            summary_writer.flush()

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # run one train_step

    # print test error
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    stop_time = time.time()
    print('The training takes %f second to finish'%(stop_time - start_time))

if __name__ == "__main__":
    main()