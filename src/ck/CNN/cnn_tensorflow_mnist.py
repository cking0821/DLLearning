#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


mnist=input_data.read_data_sets('../../../data/MNIST_data',one_hot=True)

# 定义 Weight 变量，输入 shape , 返回变量的参数。
# 使用tf.truncted_normal产生随机变量来进行初始化
def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

# 定义 bias 变量，输入shape ,返回变量的一些参数。
# 使用tf.constant常量函数来进行初始化
def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

# 定义卷积，tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，
'''
x是图片的所有参数，W是此卷积层的权重，
然后定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，
中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME。
'''
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


# 定义池化 pooling
'''
为了得到更多的图片信息，padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，
这样得到的图片尺寸没有变化，而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，
因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。
pooling 有两种，一种是最大值池化，一种是平均值池化，
本例采用的是最大值池化tf.max_pool()。
池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
tf.layers.max_pooling2d
'''
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# 图片处理

# 定义一下输入的placeholder
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

# 定义了dropout的placeholder，它是解决过拟合的有效手段
keep_drop=tf.placeholder(tf.float32)

'''
接着呢，我们需要处理我们的xs，把xs的形状变成[-1,28,28,1]，
-1代表先不考虑输入的图片例子多少这个维度，
后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
'''
x_image=tf.reshape(xs,[-1,28,28,1])

###########################################
#             建立卷积层                    #
###########################################

# 定义本层的Weight,
'''
定义第一层卷积,先定义本层的Weight,
本层我们的卷积核patch的大小是5x5，
因为黑白图片channel是1所以输入是1，输出是32个featuremap
'''
W_conv1=weight_variable([5,5,1,32])

# 定义bias，
# 它的大小是32个长度，因此我们传入它的shape为[32]
b_conv1=bias_variable([32])

'''
定义好了Weight和bias，
我们就可以定义卷积神经网络的第一个卷积层 h_conv1=conv2d(x_image,W_conv1)+b_conv1,
同时我们对h_conv1进行非线性处理，也就是激活函数来处理喽，这里我们用的是tf.nn.relu（修正线性单元）来处理，
要注意的是，因为采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，只是厚度变厚了，因此现在的输出大小就变成了28x28x32
'''
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)

# 最后我们再进行pooling的处理就ok啦，经过pooling的处理，输出大小就变为了14x14x32
h_pool1=max_pool_2x2(h_conv1)


# 接着呢，同样的形式我们定义第二层卷积，
# 本层我们的输入就是上一层的输出，本层我们的卷积核patch的大小是5x5，
# 有32个featuremap所以输入就是32，输出呢我们定为64
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

# 接着我们就可以定义卷积神经网络的第二个卷积层，这时的输出的大小就是14x14x64
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

# 最后也是一个pooling处理，输出大小为7x7x64
h_pool2=max_pool_2x2(h_conv2)

# 定义全连接层。 我们的 fully connected layer,
'''
进入全连接层时, 我们通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据,
-1表示先不考虑输入图片例子维度, 将上一个输出结果展平.
'''
#[n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

# 此时weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64， 后面的输出size我们继续扩大，定为1024
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

# 然后将展平后的h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# 如果我们考虑过拟合问题，可以加一个dropout的处理
h_fc1_drop1=tf.nn.dropout(h_fc1,keep_drop)

# 接下来我们就可以进行最后一层的构建了，好激动啊, 输入是1024，最后的输出是10个 (因为mnist数据集就是[0-9]十个类)，
# prediction就是我们最后的预测值
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

# 然后呢我们用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop1,W_fc2) + b_fc2)

##########################################
#                选优化方法                #
##########################################

# 接着呢我们利用交叉熵损失函数来定义我们的cost function
cross_entropy=tf.reduce_mean(
    -tf.reduce_sum(ys*tf.log(prediction),
    reduction_indices=[1]))

# 我们用tf.train.AdamOptimizer()作为我们的优化器进行优化，使我们的cross_entropy最小
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# 接着呢就是和之前视频讲的一样喽 定义Session
sess=tf.Session()

# 初始化变量
# tf.initialize_all_variables() 这种写法马上就要被废弃
# 替换成下面的写法:
sess.run(tf.global_variables_initializer())

# 接着就是训练数据啦，我们假定训练1000步，每50步输出一下准确率，
# 注意sess.run()时记得要用feed_dict给我们的众多 placeholder 喂数据哦.

######################################################
#                   保存
######################################################
# 搭建好了一个神经网络, 训练好了, 肯定也想保存起来, 用于再次加载.
# 那今天我们就来说说怎样用 Tensorflow 中的 saver 保存和加载吧.

# 建立神经网络当中的 W 和 b, 并初始化变量.

## Save to file
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# 替换成下面的写法:
init = tf.global_variables_initializer()

# 保存时, 首先要建立一个 tf.train.Saver() 用来保存, 提取变量.
# 再创建一个名为my_net的文件夹, 用这个 saver 来保存变量到这个目录 "my_net/save_net.ckpt".
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path: ", save_path)

"""
Save to path:  my_net/save_net.ckpt
"""


#########################################################
#                    提取
#########################################################
# 提取时, 先建立零时的W 和 b容器. 找到文件目录, 并用saver.restore()我们放在这个目录的变量.
# 先建立 W, b 的容器
tf.reset_default_graph() #加上。 使得每次恢复时模型会转化成默认的图。。因为再次读取模型时，权重名称已经发生变化，所以就会报错误
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# 这里不需要初始化步骤 init= tf.initialize_all_variables()

saver = tf.train.Saver()
with tf.Session() as sess:
    # 提取变量
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))

"""
weights: [[ 1.  2.  3.]
          [ 3.  4.  5.]]
biases: [[ 1.  2.  3.]]
"""




print ("****************888end**********")



