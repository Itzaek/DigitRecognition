#TfLearn version of DEEPMNIST
#taking from https://www.tensorflow.org/get_started/mnist/pros
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Create input object which reads data from MNIST datasets.
# Perform one-hot encoding to define the digit
#(Retrieves and creates a data set of training models and encodes it by a matrix of 1's/0's)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Creates the size of the data shape
image_width = 28
image_height = 28
number_of_classes = 10

#Create a new tensorflow session which will enable the model to train on the dataset
sess = tf.InteractiveSession()

#Define placeholders for the MNIST data set. These placeholders are supposed to show the shape of which the dataset
#will be modelled like
x = tf.placeholder(tf.float32, shape=[None,image_width*image_height])
y_ = tf.placeholder(tf.float32, shape=[None,number_of_classes])

#Define the weights and bias of the model. THese are always different for each modle and will be in each model
W = tf.Variable(tf.zeros([image_width*image_height, number_of_classes]))
b = tf.Variable(tf.zeros([number_of_classes]))

#Before you use any variables you define you have to initialize them in the tensorflow session
sess.run(tf.global_variables_initializer())

#Now we define the type of model we will be using to train this model. In this case, we will be using a softmax
#function. The function itself is a generalization of the sigmoid function to make an N outcome distribution
#In this model in particular, the weights and placefholder x are matrix multiplied and then added with the bias
#before being put into the softmax function
y = tf.nn.softmax(tf.matmul(x,W)+b)

#The loss function is used to determine how close or far the model's prediction is from the actual result. The cross
#entropy model in particular is especially strict towards predictions that were confident, but are wrong
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

#Now we start training the model
#In this exmple we sill use the steepest gradient descent (based on derivatives of a multi variable function)
#and using a step function of 0.5 (The step determines how "quickly" the model will train, the downside is skipping through overlooked losses)
learning_rate = 0.5
number_of_steps = 10000
batch_size = 100
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#Training the model
for _ in range(number_of_steps):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#Evaluate the model
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Print out the accuracy
acc_eval = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print(f"Current accuracy: %{acc_eval * 100}")
