#!/bin/python3.6

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

#Cardy Wei
#Deep Learning
#Professor Curro

np.random.seed(31415)
BATCH_SIZE = 720
NUM_BATCHES = 50000

def spirals(num_samp,noise):
    n = np.sqrt(np.random.rand(num_samp,1)) * 720 * (2*np.pi)/360
    x1 = np.cos(n)*n + np.random.rand(num_samp,1) * noise
    x2 = np.sin(n)*n + np.random.rand(num_samp,1) * noise
    spiral1 = np.hstack((x1,x2))
    spiral2 = np.hstack((-x1,-x2))
    total = np.vstack((spiral1,spiral2))
    labels = np.hstack((np.zeros(num_samp),np.ones(num_samp)))
    return total,labels,spiral1,spiral2


def f(x):
    y_hat = tf.matmul(tf.nn.elu(tf.matmul(tf.nn.elu(tf.matmul(x,w1) + b1),w2) + b2),w3) + b3
    return y_hat

w1 = tf.get_variable('w1', [2, 42], tf.float32, tf.truncated_normal_initializer())
w2 = tf.get_variable('w2', [42, 32], tf.float32, tf.truncated_normal_initializer())
w3 = tf.get_variable('w3', [32, 1], tf.float32, tf.truncated_normal_initializer())

b1 = tf.get_variable('b1', [1,42], tf.float32, tf.zeros_initializer()) 
b2 = tf.get_variable('b2', [1, 32], tf.float32, tf.zeros_initializer())   
b3 = tf.get_variable('b3', [1,1], tf.float32, tf.zeros_initializer())

x = tf.placeholder(tf.float32, [BATCH_SIZE,2])
y = tf.placeholder(tf.float32, [BATCH_SIZE,1])
y_hat = f(x)
l2norm = tf.pow(tf.norm(w1),2) + tf.pow(tf.norm(w2),2) + tf.pow(tf.norm(w3),2)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat)) + 0.001 * l2norm
optim = tf.train.AdamOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total,labels,spiral1,spiral2 = spirals(1000,.5)

for i in tqdm(range(0,NUM_BATCHES)):
    choices = np.random.choice(2000,BATCH_SIZE)
    x_np = total[choices]
    y_np = np.array(labels[choices]).reshape(BATCH_SIZE,1)
    loss_np, _ = sess.run([loss,optim],feed_dict={x:x_np, y:y_np})

xx,yy = np.meshgrid(np.arange(-15,15,0.05), np.arange(-15,15,0.05))
inputs = np.stack((xx,yy), axis=2)
y_hats = np.array([])
z = tf.placeholder(tf.float32, [inputs.shape[0],2])
y_hat = f(z)

for i in tqdm(range(inputs.shape[0])):
    if y_hats.size>0: 
        y_hats = np.hstack((y_hats,sess.run(y_hat,{z: inputs[i]})))
    else: 
        y_hats = sess.run(y_hat,{z:inputs[i]})
        
y_hats[y_hats>0] = 1
y_hats[y_hats<=0] = 0

plt.contourf(yy,xx,y_hats,alpha=0.5)
plt.plot(spiral1[0:-1,0], spiral1[0:-1,1], ".", label="Spiral 1", alpha=0.5)
plt.plot(spiral2[0:-1,0], spiral2[0:-1,1], ".", label="Spiral 2", alpha=0.5)
plt.savefig("MLHW2.pdf")
plt.show()

# What I Learned:
#     - Activation functions are important, testing out a variety of them showed me that using elu was the best
#     - Initial runs of the code showed that the number of nodes and number of layers was important
#     - Initially worked with fewer nodes and more layers gave me good results, but wasn't as efficient
#     - Finding a balance of nodes of layers is important
#     - For more precise results, lower the learning rate and increase the number of epochs for training