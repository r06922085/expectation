from agent_dir.agent import Agent
from agent_dir.lib import *
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple, MultiRNNCell
import numpy as np
import gym
import tensorflow as tf
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = str(random.randint(0,3)) #共4個gpu

    
class model(Agent):
    def __init__(self, env, args):
        self.batch_size = args.batch_size
    
        super(model,self).__init__(env)
        
        self.build()
        
        if args.train_exp:
            pass
        
        if args.test_exp:
            #you can load your model here
            print('loading trained model')
            self.load_checkpoint()
            
    def init_game_setting(self):
        self.is_first = True
        self.pre_impress_states = np.zeros((self.impress_lay_num, 2, self.batch_size, self.impress_dim))
        
    def set_parameter(self):
        print('it is setting parameters...')
        # network parameter
        self.state_dim = 80*80*1
        self.action_num = 3
        self.encoder_dim = 1024 # output dim of encoder
        self.impress_dim = 1024 # hidden dim of impress
        self.decoder_dim = self.state_dim # output dim of decoder
        self.impress_lay_num = 3
        self.games = 0
        
        # placeholder
        self.state_PH = tf.placeholder(tf.float32, [None, 80, 80, 1])
        self.pre_state_PH = tf.placeholder(tf.float32, [None, 80, 80, 1])
        
        self.pre_state_code_PH = tf.placeholder(tf.float32, [None, self.encoder_dim])
        
        self.state_impress_code_PH = tf.placeholder(tf.float32, [None, self.impress_dim])
        self.pre_state_impress_code_PH = tf.placeholder(tf.float32, [None, self.impress_dim])
        
        self.pre_impress_states_PH = tf.placeholder(tf.float32, [self.impress_lay_num, 2, None, self.impress_dim])
        
        self.is_first_in_impress_PH = tf.placeholder(tf.bool)
        
        self.pre_action_PH = tf.placeholder(tf.float32, [None, self.action_num])
        self.reward_PH = tf.placeholder(tf.float32, [None, 1])
        
        # variable between flows
        self.state = None #2D
        self.pre_state = None
        
        self.state_code = None
        self.state_impress_code = None
        
        self.pre_state_code = None
        self.pre_state_impress_code = None
        
        self.action = None
        self.pre_action = None
        
        self.impress_states = None
        self.pre_impress_states = None
        
        self.reward = None
        self.pre_reward = None
        
        # tf parameter
        self.lr = 1e-3
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
        config.gpu_options.allow_growth = True #allocate dynamically
        self.sess = tf.InteractiveSession(config=config)
        self.save_path='model_file/exp.ckpt'
        
    def encoder(self, state):
        # 2 conv layers and 1 dense layer, 80x80x1 to 512
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        
            state = tf.reshape(state, [-1,80,80,1])
            
            with tf.variable_scope('conv1'):
                net = tf.layers.conv2d(state,filters=16,kernel_size=(8, 8),strides=(4, 4),name="conv",padding='SAME')
                net = tf.nn.relu(net, name="relu")
            
            with tf.variable_scope('conv2'):
                net = tf.layers.conv2d(net,filters=16,kernel_size=(4, 4),strides=(2, 2),name="conv",padding='SAME')
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope('fc1'):
                net = tf.contrib.layers.flatten(net)
                net = tf.layers.dense(net, self.encoder_dim)
                state_code = tf.nn.relu(net, name='relu') 
        
        return state_code
        
    def impress(self, state_code, pre_impress_states, is_first_in_impress):  
        with tf.variable_scope('impress', reuse=tf.AUTO_REUSE):
            def loop_fn(time, cell_output, cell_state, loop_state):
                if cell_output is None:#time = 0
                    # initialization
                    input = state_code
                    state = state_
                    emit_output = None
                    loop_state = None
                else:
                    input = cell_output
                    emit_output = cell_output
                    state = cell_state
                    loop_state = None
                    
                elements_finished = (time >= 1)
                return (elements_finished, input, state, emit_output, loop_state)

            multirnn_cell = MultiRNNCell([LSTMCell(self.impress_dim) 
                    for _ in range(self.impress_lay_num)],  state_is_tuple=True) 
            
            if is_first_in_impress == True:
                state_ = (multirnn_cell.zero_state(self.batch_size, tf.float32))
            else:
                pre_impress_states = tf.unstack(pre_impress_states, axis=0)
                state_ = tuple(
                         [LSTMStateTuple(pre_impress_states[idx][0], pre_impress_states[idx][1])
                         for idx in range(self.impress_lay_num)])   
    
            emit_ta, states, final_loop_state = tf.nn.raw_rnn(multirnn_cell, loop_fn)
            state_impress_code = tf.transpose(emit_ta.stack(), [1, 0, 2])[0] # transpose for putting batch dimension to first dimension
            
            return state_impress_code, states
        
    def expectator(self, pre_state_impress_code, action):
        # fully connecteed, 3 layers
        with tf.variable_scope('expectator', reuse=tf.AUTO_REUSE):
            exp_input = tf.concat((pre_state_impress_code, action), axis = 1)
            
            layer1 = tf.layers.dense(inputs=exp_input, units= self.encoder_dim, activation=tf.nn.relu)
            layer2 = tf.layers.dense(inputs=layer1, units=self.encoder_dim, activation=tf.nn.relu)
            layer_f = tf.layers.dense(inputs=layer2, units=self.encoder_dim, activation=tf.nn.relu)
            
            pred_state_code = tf.layers.dense(layer_f, self.encoder_dim, activation=tf.nn.relu)
            pred_reward = tf.layers.dense(layer_f, 1) 
        
            return pred_state_code, pred_reward
        
    def planner(self, impress_code):
        #fully connecteed network
        with tf.variable_scope('planner', reuse=tf.AUTO_REUSE):
            layer1 = tf.layers.dense(impress_code, 256)
            layer2 = tf.layers.dense(layer1, 3, activation=tf.nn.relu)
        
            action = layer2
            
        #action = tf.nn.softmax(action)
        #action = tf.one_hot(action, self.action_num)
        
        return action
        
    def decoder(self, pre_state_code):
        # 1 dense layer and 2 conv layers, encoder_dim to 80x80x1
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv1_decoder'):
                net = tf.layers.dense(pre_state_code, 1600)
                net = tf.nn.relu(net, name='relu') 
                net = tf.reshape(net, [-1,10,10,16])
            
            with tf.variable_scope('conv2_decoder'):
                net = tf.layers.conv2d_transpose(net,filters=16,kernel_size=(4, 4),strides=(2, 2),name="conv",padding='SAME')
                net = tf.nn.relu(net, name="relu")
            
            with tf.variable_scope('fc1_decoder'):
                net = tf.layers.conv2d_transpose(net,filters=16,kernel_size=(8, 8),strides=(4, 4),name="conv",padding='SAME')
                net = tf.nn.relu(net, name="relu")
            
            pre_state = tf.reshape(net, [-1,80,80,1])
        
        return pre_state
        
    def build(self):
        self.set_parameter()
    
        print('it is building...')
        
        # expectator flow
        self.state_code = self.encoder(self.state_PH)
        self.state_impress_code_for_expectator_flow, self.impress_states = self.impress(self.state_code, self.pre_impress_states_PH, self.is_first_in_impress_PH)
        
        pred_state_code, pred_reward = self.expectator(self.pre_state_impress_code_PH, self.pre_action_PH)
        
        self.state_code_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                                        labels=self.state_code,
                                        predictions=pred_state_code))
        self.reward_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                                        labels=self.reward_PH,
                                        predictions=pred_reward))
        self.expectator_flow_loss = self.state_code_loss + self.reward_loss
        
        expectator_flow_var = [v for v in tf.global_variables() 
                                if ('impress' in v.name) or ('expectator' in v.name)]
        self.expectator_flow_train_op =  (tf.train.AdamOptimizer(self.lr).minimize(self.expectator_flow_loss, 
                                                                            var_list=expectator_flow_var))

        #train planner flow
        self.forward_step = 25
        
        for i in range(self.forward_step):
            if i == 0:
                self.state_impress_code = self.state_impress_code_PH
                self.total_reward = 0
                self.pre_impress_states = self.pre_impress_states_PH
            else:
                state_code = self.pred_state_code
                self.state_impress_code, self.pre_impress_states = self.impress(state_code, self.pre_impress_states, self.is_first_in_impress_PH)
                
            action = self.planner(self.state_impress_code)
            #action = tf.argmax(action, axis=1)
            #action = tf.one_hot(action, self.action_num)
            self.pred_state_code, self.pred_reward = self.expectator(self.state_impress_code, action)
            self.total_reward += self.pred_reward[0][0]
            
        self.planner_flow_loss = self.total_reward * -1
        planner_flow_var = [v for v in tf.global_variables() 
                                if ('planner' in v.name)]
        self.planner_flow_train_op = (tf.train.AdamOptimizer(self.lr).minimize(self.planner_flow_loss,
                                                                          var_list=planner_flow_var))
        
        # encoder-decoder flow
        state_code = self.encoder(self.pre_state_PH)
        pre_state = self.decoder(state_code)
        
        self.decoder_flow_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                                        labels=self.pre_state_PH,
                                        predictions=pre_state))
        decoder_flow_var = [v for v in tf.global_variables() if ('encoder' in v.name) or ('decoder' in v.name)]
        self.decoder_flow_train_op =  (tf.train.AdamOptimizer(self.lr).minimize(self.decoder_flow_loss, 
                                                                            var_list=decoder_flow_var))
                                                                            
        #run planner
        self.action = self.planner(self.state_impress_code_PH)
        #self.action = tf.argmax(action, axis=1)
        #self.action = tf.one_hot(action, self.action_num)              
                       
        #                           
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
    
    def train(self):
        print('it is training...')
        
        last_total_reward = -21
        while True:
            #variables in training
            state = None #2D
            pre_state = None
            
            state_code = None
            state_impress_code = None
            
            pre_state_code = None
            pre_state_impress_code = None
            
            action = None
            pre_action = None
            
            impress_states = None
            pre_impress_states = np.zeros((self.impress_lay_num, 2, self.batch_size, self.impress_dim))
            
            reward = None
            pre_reward = None
            
            GameOver = False
            times = 0
            self.games += 1
            total_reward = 0
            
            #first state
            #print('firt state...')
            pre_state = prepro(self.env.reset())
            pre_state_impress_code, pre_impress_states, pre_state_code = self.sess.run(
                                                     [self.state_impress_code_for_expectator_flow,
                                                      self.impress_states,
                                                      self.state_code],
                                                     feed_dict = {self.state_PH: pre_state[np.newaxis, :],
                                                                  self.pre_impress_states_PH: pre_impress_states,
                                                                  self.is_first_in_impress_PH: True})
           
            #first action
            pre_action = np.zeros((self.batch_size, self.action_num))
            pre_action[0][random.randint(0,self.action_num-1)] = 1
            
            #second state
            state, reward, GameOver, _ = self.env.step(np.argmax(pre_action[0]))
            reward = np.array([reward])
            state = prepro(state)
    
            #enter to the loop
            while not GameOver:
                #train expectator
                #print('train expectator...')
                times += 1
               
                feed = {
                    self.state_PH: state[np.newaxis, :],
                    self.pre_state_impress_code_PH: pre_state_impress_code,
                    self.pre_action_PH: pre_action,
                    self.reward_PH: np.array(reward)[np.newaxis, :],
                    self.pre_impress_states_PH: pre_impress_states,
                    self.is_first_in_impress_PH: False
                    }
                _, expectator_loss, state_impress_code, impress_states, state_code, state_code_loss, reward_loss  = self.sess.run([
                                                                        self.expectator_flow_train_op,
                                                                        self.expectator_flow_loss,
                                                                        self.state_impress_code_for_expectator_flow,
                                                                        self.impress_states,
                                                                        self.state_code,
                                                                        self.state_code_loss,
                                                                        self.reward_loss],
                                                                        feed_dict = feed)
                                                                        
                #print('state_code_loss: ', state_code_loss)
                #print('reward_loss: ', reward_loss)
               
                #train planner
                #print('train planner...')
                planner_loss, _ = self.sess.run([self.planner_flow_loss,
                                                 self.planner_flow_train_op], 
                                                 feed_dict = {self.state_impress_code_PH: state_impress_code,
                                                              self.pre_impress_states_PH: pre_impress_states,
                                                              self.is_first_in_impress_PH: False})
    
                #train encoder-decoder
                #print('train decoder...')
                encoder_decoder_loss, _ = self.sess.run([self.decoder_flow_loss, self.decoder_flow_train_op], 
                                              feed_dict = {self.pre_state_PH: pre_state[np.newaxis, :]})
                    
                #run planner
                #print('run planner...')
                action = self.sess.run(self.action,
                                       feed_dict = {self.state_impress_code_PH: state_impress_code})
   
                #play the game of this step                       
                pre_state = state
                
                rand_seed = random.randint(0,42)
                
                if (last_total_reward+21) < rand_seed:
                    for i in range(self.action_num):
                        action[0][i] = 0
                    action[0][random.randint(0,self.action_num-1)] = 1
                
                state, reward, GameOver, _ = self.env.step(np.argmax(action[0], axis=0))
                total_reward += reward
                state = prepro(state)
                
                #prepare for next state
                pre_state_code = state_code
                pre_state_impress_code = state_impress_code
                pre_impress_states = impress_states
                reward = np.array([reward])
                pre_reward = reward
                pre_action = action
                
            last_total_reward = total_reward    
                
            print('expectator_loss:{0:.3}|planner_loss:{1:.3}|encoder-decoder_loss:{2:.3}|times: {3}'
                .format(expectator_loss, planner_loss, encoder_decoder_loss, times))
                    
            print('games{0}|reward{1}'.format(self.games, total_reward))        
            
            self.save()    
              
                
        
    def make_action(self, observation, test=True):
        state = prepro(observation)
        
        state_impress_code, self.pre_impress_states = self.sess.run([self.state_impress_code_for_expectator_flow,
                                                                     self.impress_states],
                                           feed_dict = {self.state_PH: state[np.newaxis, :],
                                                        self.is_first_in_impress_PH: self.is_first,
                                                        self.pre_impress_states_PH: self.pre_impress_states})
        
        action = self.sess.run(self.action,
                                feed_dict = {self.state_impress_code_PH: state_impress_code})
                                
        self.is_first = False
        
        #print(np.argmax(action[0][0], axis=0))
        print(np.argmax(action[0], axis=0))
        return np.argmax(action[0], axis=0)
        
    def save(self):
        print('Saving checkpoint...')
        self.saver.save(self.sess, self.save_path)

    def load_checkpoint(self):
        print('Loading checkpoint')
        self.saver.restore(self.sess, self.save_path)
        