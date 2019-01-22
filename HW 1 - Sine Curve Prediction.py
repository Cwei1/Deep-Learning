#!/bin/python3.6

#Cardy Wei
#Professor Curro
#Deep Learning Assignment 1

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

M = 8
BATCH_SIZE = 32
NUM_BATCHES = 300
num_samp = 50

class Data(object):
    def __init__(self):
        sigma = 0.1
        np.random.seed(31415)

        self.index = np.arange(num_samp)        
        self.x = np.random.uniform(size=(num_samp,1))
        self.y = np.sin(2*np.pi*self.x) + sigma * np.random.normal()
        plt.plot(self.x,self.y,'o')
        
    def get_batch(self):
        choices = np.random.choice(self.index, size=(BATCH_SIZE))

        return self.x[choices].flatten() , self.y[choices].flatten()


def f(x):
    w = tf.get_variable('w', [1, M], tf.float32, tf.random_normal_initializer())
    b = tf.get_variable('b', [], tf.float32, tf.zeros_initializer())
    mu = tf.get_variable('mu', [M,1], tf.float32, tf.random_normal_initializer())
    sigma = tf.get_variable('sigma', [M, 1], tf.float32, tf.random_normal_initializer())
    phi = tf.exp(-1*tf.pow((x-mu),2)/tf.pow(sigma,2))
    return tf.squeeze(tf.matmul(w,phi) + b)

x = tf.placeholder(tf.float32, [BATCH_SIZE])
y = tf.placeholder(tf.float32, [BATCH_SIZE])
y_hat= f(x)

loss = tf.reduce_mean(tf.pow(y_hat - y, 2))
optim = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
data = Data()

for _ in tqdm(range(0, NUM_BATCHES)):
    x_np, y_np = data.get_batch()
    loss_np, _ = sess.run([loss, optim], feed_dict={x: x_np, y: y_np})

var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

w0 = sess.run(var[0])
b0 = sess.run(var[1])
mu0 = sess.run(var[2])
sigma0 = sess.run(var[3])

print("Parameter estimates:")
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(
        var.name.rstrip(":0"),
        np.array_str(np.array(sess.run(var)).flatten(), precision=3))

xvals = np.linspace(0,1,num=10000)
xvals2 = np.linspace(0,1,num=100)
yhat = []
for i in xvals:
    yhat.append((w0 @ np.exp(-1*np.square(i-mu0)/np.square(sigma0)) + b0).flatten())

normalx = np.arange(0.0, 1.0, 0.01)
normaly = np.sin(2*np.pi*normalx)

plt.plot(normalx,normaly)

plt.plot(xvals,yhat, linestyle='dotted', color='red')
plt.title("Sinewave with Noise, Sinewave, and Predicted Curve")
plt.ylabel("y")
plt.xlabel("x")
plt.savefig("test.svg")
plt.show()

for i in range(M):
	plt.plot(xvals2,np.exp(-1*np.square(xvals2-np.repeat(mu0[i], 100))/np.square(np.repeat(sigma0[i],100))))

plt.title("Gaussians used for Prediction")
plt.ylabel("y")
plt.xlabel("x")
plt.savefig("test2.svg")
plt.show()