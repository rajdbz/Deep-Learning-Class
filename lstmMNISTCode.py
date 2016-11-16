import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)  # call mnist function

learningRate = 0.001
trainingIters = 200000
batchSize = 128
displayStep = 10

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 32  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know

# tf Graph input
x = tf.placeholder('float', [None, nSteps, nInput])  # input
y = tf.placeholder('float', [None, nClasses])  # label

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(0, nSteps, x)  # configuring so you can get it as needed for the 28 pixels

    # Using RNN
    #lstmCell = rnn_cell.BasicRNNCell(nHidden)

    # Using LSTM
    #lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)  # find which lstm to use in the documentation
    lstmCell = rnn_cell.BasicRNNCell(nHidden)

    outputs, states = rnn.rnn(lstmCell, x, dtype=tf.float32)  # for the rnn where to get the output and hidden state

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# Add a scalar summary for the snapshot of accuracy and loss
accuracy_summary = tf.scalar_summary(tags='Accuracy', values=accuracy)
loss_summary = tf.scalar_summary(tags='Loss', values=cost)

# Build the summary operation based on the TF collection of Summaries.
summary_merged = tf.merge_all_summaries()

# Add the variable initializer Op.
init = tf.initialize_all_variables()

result_dir = '/Users/rajdbz/Desktop/Coursework/ELEC677/Assignment2/result_lstm/rnn'

with tf.Session() as sess:
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    train_summary_writer = tf.train.SummaryWriter(result_dir + '/training', sess.graph)
    test_summary_writer = tf.train.SummaryWriter(result_dir + '/test')

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    testData = mnist.test.images[:test_len].reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels[:test_len]

    step = 1
    while step * batchSize < trainingIters:
        batchX, batchY = mnist.train.next_batch(batch_size=batchSize)  # mnist has a way to get the next batch

        batchX = batchX.reshape((batchSize, nSteps, nInput))

        if step % displayStep == 0:
            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
            loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
            print("Iter " + str(step * batchSize) + ", Mini batch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            '''
            Training summary
            '''
            # obtain full summary (i.e. accuracy and loss)
            train_str = sess.run(summary_merged, feed_dict={x: batchX, y: batchY})
            train_summary_writer.add_summary(train_str,step)
            train_summary_writer.flush()
            '''
            Test summary
            '''
            test_str = sess.run(summary_merged,feed_dict={x: testData, y: testLabel})
            test_summary_writer.add_summary(test_str,step)
            test_summary_writer.flush()

        # Update the train step
        train_step = sess.run(optimizer, feed_dict={x: batchX, y: batchY})

        step += 1

    print('Optimization finished')
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: testData, y: testLabel}))

