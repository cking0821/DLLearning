# 使用tensorflow来解决MNIST手写体数字识别问题
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
input_node = 784  # 输入层的节点数，图片的像素
output_node = 10  # 输出层的节点数，0-9的类别数

# 配置神经网络的参数
layer_node = 500  # 隐藏层节点数
batch_size = 100  # 一个训练batch中数据个数，数字越小，越接近随机梯度下降，越大，越接近梯度下降
learning_rate_base = 0.8  # 基础的学习率
learning_rate_decay = 0.99  # 学习率的衰减率
regularization_rate = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
training_steps = 30000  # 训练轮数
moving_average_decay = 0.99  # 滑动平均衰减率


# 给定神经网络的输入与所有参数，计算神经网络的前向通达的传播结果
def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
        return tf.matmul(layer1, weight2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)
        # 训练模型的过程


def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')

    # 生成隐藏层的参数
    weight1 = tf.Variable(tf.truncated_normal([input_node, layer_node], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer_node]))

    # 生成输出层的参数
    weight2 = tf.Variable(tf.truncated_normal([layer_node, output_node], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))

    # 计算当前参数下神经网络前向通道的结果，不使用滑动平均值
    y = inference(x, None, weight1, biases1, weight2, biases2)
    # 代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)
    # 给定滑动平均衰减率与训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)
    # 计算交叉熵作为刻画预测值与真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    # 计算模型的正则化损失
    regularization = regularizer(weight1) + regularizer(weight2)
    # 总损失
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                               mnist.train.num_examples / batch_size,
                                               learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据就要通过BP更新神经网络的参数以及每个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播结果是否挣正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 计算模型的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 迭代的训练神经网络
        for i in range(training_steps):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using average model is %g" % (i, validate_acc))
            # 产生这一轮使用的batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 测试结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training steps ,test accuracy using average model is %g' % (training_steps, test_acc))


# 主程序入口
def main(argv=None):
    # 声明处理MNIST 数据集的类，这个类在初始化时会自动下载
    mnist = input_data.read_data_sets("../../../data/MNIST_data", one_hot=True)
    train(mnist)


# Tensorflow 提供的一个主程序入口，tf.app.run 会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()