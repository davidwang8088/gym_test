# -*- coding: utf-8 -*-
import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from gym import wrappers


#initialize hyperparameters
H = 20  #hidden neurons
render = True  #Decide whether render or not
EPISODE = 10000
BATCH_SIZE = 32
REPLAY_SIZE = 10000 #max number of restoring clips
TEST = 10
STEP_WIN = 300 #if run over STEP_WIN, then win the game.
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
LEARNING_RATE = 0.0002
UPDATE_TARGET_MODEL_STEP = 200

#initializing virables
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env,'/tmp/cartpole-experiment-1',force = True)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
total_step = 0
Replay_list = [] #store every s_t, a_t, r_t,1 s_t+1 tuple
discount = 0.9
epsiron = INITIAL_EPSILON
loss_list = [] # For plotting loss
episode_list = [] #For plotting loss


# feed forward graph construction
W1 = tf.Variable(tf.random_normal([input_dim,H]), tf.float64)
b1 = tf.Variable(tf.zeros([H]), tf.float64)
W2 = tf.Variable(tf.random_normal([H,output_dim]), tf.float64)
b2 = tf.Variable(tf.zeros([output_dim]), tf.float64)
s = tf.placeholder(tf.float32, [None,input_dim])
h_layer = tf.nn.relu(tf.matmul(s,W1) + b1)
q_value = tf.matmul(h_layer, W2) + b2


#training graph construction
y = tf.placeholder(tf.float32, [None])
action = tf.placeholder(tf.float32, [None, output_dim])
q_action_debug = tf.mul(q_value,action)
q_action = tf.reduce_sum(tf.mul(q_value, action), 1)
loss = tf.reduce_mean(tf.square(y - q_action))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
train = optimizer.minimize(loss)

rList= []

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    # Target_Q notwork virables
    W1_target = sess.run(W1)
    b1_target = sess.run(b1)
    W2_target = sess.run(W2)
    b2_target = sess.run(b2)

    for episode in range(EPISODE):

        cur_obsv = env.reset()
        reward_all = 0

        #every loss in the episode
        loss_episode = 0

        # feed forward and store the s,a,r,s+1,d in the replay list
        for step in range(STEP_WIN):
            total_step += 1
            #env.render()
            cur_obsv_t = cur_obsv.reshape(1, input_dim)#将一维向量observation(4,)转化为1×4的二维矩阵(1,4)
            Q_value = sess.run(q_value, feed_dict = {s: cur_obsv_t})[0] # Evaluate the feed fordward graph

            #epsiron greedy policy to choose action
            if np.random.random() > epsiron:
                cur_action = np.argmax(Q_value)
            else:
                cur_action = env.action_space.sample()

            epsiron -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000.

            #step forward
            next_obsv, cur_reward, done, _ = env.step(cur_action)

            #put the action into the one-hot style   注意：action = 0 的onehot值是[1 0]
            one_hot_action = np.zeros(output_dim)
            one_hot_action[cur_action] = 1

            #restore current s, a, r, s+1, d in the replaylist
            Replay_list.append([cur_obsv, one_hot_action, cur_reward, next_obsv, done])


            # Train the nn model
            if len(Replay_list) > REPLAY_SIZE:
                Replay_list.pop(0)

            # Training the model if Replay_list is bigger than batch_size
            if len(Replay_list) > BATCH_SIZE:
                #sample from replay list to train the nn model
                batch_list = random.sample(Replay_list, BATCH_SIZE)
                #put the list into (Batch_size, 5) array, '5' is the shape of list [st,at,rt,st+1,done], in which eyery element is "object" type
                nbatch = np.vstack(batch_list)
                #get the data by column, which is a (Batch_size, ) 1d array, then another vstack change the 1d array into (Batch_size, object.size)
                batch_s = np.vstack(nbatch[:,0]) #(Batch_size, st.size)= (32, 5)
                batch_a = np.vstack(nbatch[:,1])
                batch_r = np.vstack(nbatch[:,2])
                batch_s1 = np.vstack(nbatch[:,3])
                batch_done = np.vstack(nbatch[:,4])

                y_list = []

                # print 'W1:\n',sess.run(W1)
                # print 'W1_plus:\n', sess.run(W1_plus)
                # update parameters every 500 steps

                #calculate Q_target values
                feed_dict = {s:batch_s1}
                feed_dict.update(zip([W1,b1,W2,b2], [W1_target,b1_target,W2_target,b2_target]))
                Q_targt = sess.run(q_value,feed_dict = feed_dict)
                #calculate fake ground turth Q value, y_i
                for i in range(BATCH_SIZE):
                    if batch_done[i][0]:
                        y_list.append(batch_r[i][0])
                    else:
                        # if not the final state, y_i plus discouted max qvalue_t+1
                        y_list.append(batch_r[i][0]+ discount * np.max(Q_targt[i]))
                y_array = np.hstack(y_list)

                sess.run(train, feed_dict = {y:y_array, s:batch_s, action:batch_a})

                #Update target model parameters
                if total_step % UPDATE_TARGET_MODEL_STEP == 0:
                    W1_target = sess.run(W1)
                    b1_target = sess.run(b1)
                    W2_target = sess.run(W2)
                    b2_target = sess.run(b2)

                if total_step % 100 == 0:
                    debug_loss = sess.run(loss, feed_dict={y: y_array, s: batch_s, action: batch_a})
                    print debug_loss
                    loss_list.append(debug_loss)

            cur_obsv = next_obsv
            if done:
                break


        #test
        if episode % 100 == 0:
            total_reward = 0
            total_q = 0
            for i in range(TEST):
                test_obsv = env.reset()
                for j in range(STEP_WIN):
                    env.render()
                    test_obsv_t = test_obsv.reshape(1, input_dim)
                    test_q = sess.run(q_value, feed_dict = {s: test_obsv_t})[0]
                    test_action = np.argmax(test_q)
                    test_obsv, test_reward, test_done, info = env.step(test_action) # put the next observation to test_obsv_t directly
                    total_reward += test_reward
                    total_q += np.max(test_q)

                    if test_done:
                        break

            avg_reward = total_reward / TEST
            avg_q = total_q / TEST
            print 'episode: ', episode, 'Evaluation Average Reward:', avg_reward
            # print 'episode: ', episode, 'Evaluation Average QValue:', avg_q, '!!!!!!!!!'
            if avg_reward >= 200:
                break

env.close()
gym.upload('/tmp/cartpole-experiment-1', api_key='sk_X8YI3v5xSVeqhMcGVUAYLw')
# plot the loss curves
plt.plot(loss_list)
plt.ylabel('Loss')
plt.show()
