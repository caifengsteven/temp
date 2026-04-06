
# coding: utf-8

# 在看一篇介绍增强学习基本原理的博客（ http://karpathy.github.io/2016/05/31/rl/ ）的时候想到，既然可以通过训练机器学习打乒乓球，那能否训练机器去学习如何交易呢？后来也看到有些其他关于深度学习的教材提到了使用增强学习的算法来实现自动交易。因此，自己就参考了TensorLayer里关于增强学习的例子（ https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_atari_pong.py ）并结合到股票交易数据的特殊性进行了一下尝试。

# 这里选择青岛海尔这只票，选它的原因是因为刚好买了这只股票，发现它的贝塔值特别大，经常波动超过百分三四，所以觉得可能波动大会带来交易机会。由于优矿个人版不支持tensorflow平台，下面所有的计算都是在我本机上完成，并且不考虑交易费用以及冲击成本的影响。

# In[ ]:

qdhr = DataAPI.MktEqudAdjGet(tradeDate=u"",secID=u"",ticker=u"600690",beginDate=u"",endDate=u"",isOpen="",field=u"",pandas="1")
qdhr.to_csv('qdhr.csv', index = False, encoding = 'GBK')


# In[ ]:

import os
import pandas as pd
import tensorflow as tf
import tensorlayer as tl


import time
import numpy as np

os.chdir('C:\\Users\\Wonder\\Downloads')
#os.chdir('E:\\Downloads')
stockPrice = pd.read_csv('qdhr.csv', encoding = 'GBK')
stockPrice = stockPrice[stockPrice.isOpen == 1]
stockPrice['rtn1'] = stockPrice.closePrice/stockPrice.openPrice - 1     #开盘价到收盘价的收益
stockPrice['rtn2'] = stockPrice.openPrice/stockPrice.preClosePrice - 1  #今开盘到昨收盘的收益
stockPrice['rtn3'] = stockPrice.closePrice/stockPrice.preClosePrice - 1 #今收盘到昨收盘的收益


# 将所有的数据分成两部分，其中80%为训练数据，20%为测试数据。由于仅仅是做一个简单的尝试，没有将训练数据再划出验证数据集，就是暂时不考虑超参数的调优问题。
# 设计的神经网络模型的结构如下：
# （1）、输入为过去10天的价格信息，这里只挑选了每日收盘价和日换手率这两个指标
# （2）、中间只有一个隐藏层，隐藏层有50个节点，激活函数使用Leaky Relu
# （3）、输出为各个操作对应的概率，实际上这里应该只需要输出一个多仓的概率，由于最开始考虑会有多种操作，选择了使用softmax作为最后的激活函数。

# In[ ]:

trLen = int(len(stockPrice)*0.8)
trainData = stockPrice[:trLen]
testData = stockPrice[trLen:]
          
lookback = 10
nFactor = 2

to_train = True


# hyperparameters
D = lookback*nFactor
H = 50
batch_size = 1
step_size = 10
learning_rate = 1e-4
gamma = 0.95
decay_rate = 0.99
# resume = True         # load existing policy network
model_file_name = "model_rl_strat"
np.set_printoptions(threshold=np.nan)


prev_pos = 0
running_reward = None
reward_cum = 1
episode_number = 0

# observation for training and inference
t_states = tf.placeholder(tf.float32, shape=[None, D])
# policy network
network = tl.layers.InputLayer(t_states, name='input')
network = tl.layers.DenseLayer(network, n_units=H, act= lambda x : tl.act.lrelu(x, 0.2), name='hidden')
network = tl.layers.DenseLayer(network, n_units=2, name='output')
probs = network.outputs
sampling_prob = tf.nn.softmax(probs)

t_actions = tf.placeholder(tf.int32, shape=[None])
t_discount_rewards = tf.placeholder(tf.float32, shape=[None])
loss = tl.rein.cross_entropy_reward_loss(probs, t_actions, t_discount_rewards)
train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)


# 由于数据量相对有限，这里是逐日开始进行10天的交易，作为一个回合训练模型，因此数据会被复用，在训练的时候会出现预测时使用了未来数据的问题。不过在测试时没有复用数据。

# In[ ]:

if to_train:   
    
    with tf.Session() as sess:
        tl.layers.initialize_global_variables(sess)
        # if resume:
        #     load_params = tl.files.load_npz(name=model_file_name+'.npz')
        #     tl.files.assign_params(sess, load_params, network)
        #tl.files.load_and_assign_npz(sess, model_file_name+'.npz', network)
        network.print_params()
        network.print_layers()
    
        start_time = time.time()
        game_number = 0
        xs, ys, disR = [], [], []

        for k in range(lookback, trLen - step_size):
            rs = []
            for p in range(step_size):
                dataLoc = k + p
                
                x = trainData.iloc[(dataLoc - lookback):dataLoc][['closePrice', 'turnoverRate']]
                x.closePrice = (x.closePrice - x.closePrice.min())/(x.closePrice.max() - x.closePrice.min())
                
                x = x.values.reshape(1, D)
                
        
                prob = sess.run(
                    sampling_prob,
                    feed_dict={t_states: x})
                
                # action. 0: void  1: close
                action = np.random.choice([0,1], p=prob.flatten())
                if prev_pos == 0:
                    if action == 0:
                        reward = 0
                    else:
                        reward = trainData.iloc[dataLoc].rtn1
                else:
                    if action == 0:
                        reward = trainData.iloc[dataLoc].rtn2
                    else:
                        reward = trainData.iloc[dataLoc].rtn3
                prev_pos = action
                reward_cum *= (1 + reward)
                xs.append(x)            # all observations in an episode
                ys.append(action)   # all fake labels in an episode (action begins from 1, so minus 1)
                rs.append(reward)       # all rewards in an episode
                episode_number += 1
                
                if reward != 0:
                    print(('episode %d: game %d took %.5fs, reward: %f' %
                                (episode_number, game_number,
                                time.time()-start_time, reward)),
                                ('' if reward == -1 else ' !!!!!!!!'))
                    start_time = time.time()
            
            
            game_number += 1
                
            
            
            rs = tl.rein.discount_episode_rewards(np.asarray(rs), gamma, mode = 1)
            rs -= np.mean(rs)
            if np.std(rs) > 0:
                rs /= np.std(rs)
            disR.extend(rs)
            if game_number % batch_size == 0:
                print('batch over...... updating parameters......')
                epx = np.vstack(xs)
                epy = np.asarray(ys)
                epr = np.asarray(disR)
                sess.run(
                    train_op,
                    feed_dict={
                        t_states: epx,
                        t_actions: epy,
                        t_discount_rewards: epr})
        
                xs, ys, disR = [], [], []
    
            reward_cum = 1
            prev_pos = 0
            if game_number % (batch_size * 50) == 0:
                tl.files.save_npz(network.all_params, name=model_file_name+'.npz')
        
        tl.files.save_npz(network.all_params, name=model_file_name+'.npz')


# 利用训练好的数据在测试数据上进行交易，同时新的交易数据也会用来更新模型，并选择买入并持有作为对照基准。由于增强学习模型在进行决策时是根据概率进行抽样而不是选择最大概率的那个，导致结果会出现一定的随机性，有的结果会好于基准，有的会表现非常一般,所以如何去衡量增强学习算法的有效性，我还是有些困惑。当然青岛海尔这只票本身的基本面很好，历史表现也非常不错。

# In[ ]:

with tf.Session() as sess:
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess, model_file_name+'.npz', network)
    
    buyholding = []
    backtestRet = []
    testDate = []
    
    rs = []
    xs, ys, disR = [], [], []
    for k in range(lookback, len(testData)):
        dataLoc = k
        buyholding.append(testData.iloc[dataLoc].rtn3)
        testDate.append(testData.iloc[dataLoc].tradeDate)
        x = testData.iloc[(dataLoc - lookback):dataLoc][['closePrice', 'turnoverRate']]
        x.closePrice = (x.closePrice - x.closePrice.min())/(x.closePrice.max() - x.closePrice.min())
        
        x = x.values.reshape(1, D)
        

        prob = sess.run(
            sampling_prob,
            feed_dict={t_states: x})
        
        if prob.flatten()[1] > 0.5:
            print(testData.iloc[dataLoc].tradeDate, testData.iloc[dataLoc].rtn3)
        
        # action. 0: void  1: close
        action = np.random.choice([0,1], p=prob.flatten())
        if prev_pos == 0:
            if action == 0:
                reward = 0
            else:
                reward = testData.iloc[dataLoc].rtn1
        else:
            if action == 0:
                reward = testData.iloc[dataLoc].rtn2
            else:
                reward = testData.iloc[dataLoc].rtn3
        backtestRet.append(reward)
        prev_pos = action
        reward_cum *= (1 + reward)
        xs.append(x)            # all observations in an episode
        ys.append(action)   # all fake labels in an episode (action begins from 1, so minus 1)
        rs.append(reward)       # all rewards in an episode
        
        if (k - lookback + 1) % step_size == 0:
            rs = tl.rein.discount_episode_rewards(np.asarray(rs), gamma, mode = 1)
            rs -= np.mean(rs)
            if np.std(rs) > 0:
                rs /= np.std(rs)
            disR.extend(rs)
            rs = []
        
        if (k - lookback + 1) % (step_size*batch_size) == 0:
            epx = np.vstack(xs)
            epy = np.asarray(ys)
            epr = np.asarray(disR)
            
            print(str(k - lookback + 1) + ' days passed...... updating parameters......')
            disR -= np.mean(disR)
            if np.std(disR) > 0:
                disR /= np.std(disR)
            sess.run(
                train_op,
                feed_dict={
                    t_states: epx,
                    t_actions: epy,
                    t_discount_rewards: epr})
    
            xs, ys, disR = [], [], []
            
rets = np.c_[backtestRet, buyholding]  
retsDF = pd.DataFrame(rets)     
retsDF.index = testDate
retsDF.columns = ['bktest', 'buyholding']
(retsDF + 1).cumprod().plot()


# In[ ]:

retsDF = pd.read_csv('retsDF.csv')
retsDF.set_index(retsDF.columns[0], inplace = True)
retsDF.index.names = ['Date']
(retsDF + 1).cumprod().plot()

