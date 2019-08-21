import os
import copy

import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import cv2
import scipy.signal
import tensorflow.keras as keras
import tensorflow.keras.backend as k

# import local classes
from AC_Network import AC_Model
from Worker import Worker, WorkerThread


class Coordinator:
    def __init__(self, global_model, local_model, workers, plot, num_envs, num_epocs, batches_per_epoch, batch_size, gamma, model_save_path, anneling_steps):
        self.global_model = global_model
        self.local_model = local_model
        self.model_save_path = model_save_path
        self.workers = workers
        self.num_envs = num_envs
        self.num_epocs = num_epocs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self._last_batch_loss = 0
        self.plot = plot   
        self.total_steps = 0
        self.pre_train_steps = 0
        self._currentE = 1.0
        self.current_prob = 0.0
        self.anneling_steps = anneling_steps
        self._current_annealed_prob = 1.0
        self._train_data = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007, epsilon=1e-5, clipnorm=.50)

    # Annealing entropy to encourage convergence later: 1.0 to 0.01
    def _current_entropy(self):
        startE = .30
        endE = 0.01 

        # Final chance of random action
        stepDrop = (startE - endE) / self.anneling_steps

        if self.total_steps % 256 == 0:
            print("Current entropy is: " + str(self._current_annealed_prob))

        if self._current_annealed_prob >= endE:
            self._current_annealed_prob -= stepDrop
            return self._current_annealed_prob
        else:
            return 0.01 
    
    # STD Mean normalization
    def _normalize(self, x, clip_range=[-100.0, 100.0]):
        norm_x = tf.clip_by_value((x - tf.reduce_mean(x)) / tf.math.reduce_std(x), min(clip_range), max(clip_range))
        return norm_x

    def _clip_by_range(self, x, clip_range=[-50.0, 50.0]):
        clipped_x = tf.clip_by_value(x, min(clip_range), max(clip_range))
        return clipped_x
            
    # Annealing temperature scales from .1 to 1.0
    def _keep_prob(self):
        keep_per = lambda: (1.0 - self._currentE) + 0.1
        startE = 1.0
        endE = 0.1 # Final chance of random action
        pre_train_steps = self.pre_train_steps # Number of steps used before anneling begins
        total_steps = self.total_steps # max steps ones can take
        stepDrop = (startE - endE) / self.anneling_steps

        if self._currentE >= endE and total_steps >= pre_train_steps:
            
            self._currentE -= stepDrop
            p = keep_per()

            if self.total_steps % 256 == 0:
                print("Current temp is: " + str(p))
            
            self.current_prob = p
            return p

        else:
            self.current_prob = 1
            return 1.0      
    
    # Convert numbered actions into one-hot formate [0 , 0, 1, 0, 0]
    def dense_to_one_hot(self, labels_dense, num_classes = 7):
        labels_dense = np.array(labels_dense)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        labels_one_hot = labels_one_hot.tolist()
        return labels_one_hot

    # Update dropout 
    def update_dropout_and_refresh_params(self):
        state_shape = self.global_model.state_shape

        new_global_model = AC_Model(state_shape=state_shape, is_training=True, num_actions=self.global_model.num_actions, dropout_rate=1-self.current_prob)
        new_global_model(tf.convert_to_tensor(np.random.random((1, state_shape))))
        new_global_model.set_weights(self.global_model.get_weights())
        self.global_model = new_global_model
        
        new_local_model = AC_Model(state_shape=state_shape, is_training=True, num_actions=self.local_model.num_actions, dropout_rate=1-self.current_prob)
        new_local_model(tf.convert_to_tensor(np.random.random((1, state_shape))))
        new_local_model.set_weights(self.global_model.get_weights())
        self.local_model = new_local_model


    # Used to copy over global variables to local network 
    def refresh_local_network_params(self):
        global_weights = self.global_model.get_weights()
        self.local_model.set_weights(global_weights)
       
    # pass a tuple of (batch_states, batch_actions,batch_rewards)
    def loss(self):
        prob = self.current_prob
        batch_states, actions, rewards, batch_advantages, _ = self._train_data
        actions_hot = tf.one_hot(actions, self.global_model.num_actions, dtype=tf.float64)
        logits, action_dist, values = self.global_model.call(tf.convert_to_tensor(np.vstack(np.expand_dims(batch_states, axis=1)), dtype=tf.float64), keep_p=prob)
        rewards = tf.Variable(rewards, name="rewards", dtype=tf.float64, trainable=False)
        
        # Remove the extra dims
        values = tf.squeeze(values)


        # Calculate and then mean-std normalize advantages
        advantages = rewards - values
        advantages = self._normalize(advantages)

        # Entropy bonus
        # Entropy: (1 / n) * - ∑ P_i * Log (P_i)
        entropy = tf.reduce_mean(self.global_model.softmax_entropy(action_dist))

        # Value loss
        # "MSE" (1 / n) * ∑[V(i) - R_i]^2
        squared_error = tf.square(values - rewards)
        value_loss = tf.reduce_mean(squared_error)

        # Policy loss
        # (1 / n) * ∑ * -log π(a_i|s_i) * A(s_i, a_i) 
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=actions_hot, logits=logits) 
        policy_loss = tf.reduce_mean(tf.stop_gradient(advantages) * neg_log_prob)

        # Final total loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
        self._last_batch_loss = total_loss = policy_loss - entropy * self.global_model.entropy_coef + value_loss * self.global_model.value_function_coeff

        return total_loss
      
    def train(self, train_data):
   
        self._train_data = train_data

        # Apply Gradients
        params = self.global_model.trainable_variables
        self.optimizer.minimize(self.loss, var_list=params)

        self.collect_stats("LOSS", self._last_batch_loss.numpy())

    # request access to collector and record stats
    def collect_stats(self, key, value):
        while self.plot.busy_notice():
            continue
        self.plot.stop_request()
        self.plot.collector.collect(key, value)
        self.plot.continue_request()


    # Produces reversed list of discounted values
    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # Produces reversed list of bootstrapped discounted r
    def rewards_discounted(self, rewards, gamma, bootstrap):
        
        discounted_rewards = []
        reward_sum = bootstrap
        
        for reward in rewards[::-1]:  
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        result = discounted_rewards[::-1]
        return result

    # for debugging processed images
    def displayImage(self, img):
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        save_path = ".\Model\\" + "state_vars.txt"

        try: 
            if (os.path.exists(save_path)):
                with open(save_path, 'r') as f:
                    s = f.read()
                    self.total_steps = int("".join(s.split()))
                    f.close()
        except:
            print("No additional saved state variables.")
        
        self.plot.start()
        
        try:
            # Main training loop
            for _ in range(self.num_epocs):

                # ready workers for the next epoc, sync with live plot
                while self.plot.busy_notice():
                    continue
                
                self.plot.stop_request()
                
                for worker in self.workers:
                    worker.reset()

                self.plot.continue_request()
        

                # loop for generating a training session a batch at a time
                for mb in range(self.batches_per_epoch):
                    
                    self.total_steps += 1   
                    
                    # Copy global network over to local network, update dropout
                    prob = self._keep_prob()
                    # self.update_dropout_and_refresh_params() 
                    self.refresh_local_network_params()

                    # Send workers out to threads
                    threads = []
                    for worker in self.workers:
                        threads.append(WorkerThread(target=worker.run, args=([prob])))

                    # Start the workers on their tasks
                    for thread in threads:
                        thread.start()
                        
                    batches = []
                    
                    # Wait foreach worker to finish and return their batch
                    for thread in threads:
                        batch = thread.join()
                        batches.append(batch)

                    all_rewards = np.array([])
                    all_states = np.array([])
                    all_actions = np.array([])
                    all_values = np.array([])
                    all_advantages = np.array([])

                    # Calculate discounted rewards for each environment
                    for env in range(self.num_envs):
                        done = False
                        batch_rewards = []
                        batch_observations = []
                        batch_states = []
                        batch_actions = []
                        batch_values = []
                        batch_advantages = []

                        mb = batches[env]
                        
                        # Empty batch
                        if mb == [] or mb == None:
                            continue


                        # For every step made in this env for this particular batch
                        done = False
                        value = None
                        observation = None

                        for step in mb:
                            (state, observation, reward, value, action, done, logits) = step
                            batch_actions.append(action)
                            batch_rewards.append(reward)
                            batch_observations.append(observation)
                            batch_states.append(state)
                            batch_values.append(value)

                        
                        # If we reached the end of an episode or if we filled a batch without reaching termination of episode
                        # we boot-strap the final rewards with the V_s(last_observation)

                        # Discounted bootstraped rewards formula:
                        # G_t == R + γ * V(S_t'):

                        # Advantage
                        # δ_t == R + γ * V(S_t') - V(S_t): 
                        
                        # Or simply... 
                        # δ_t == G_t - V: 
                        # 
                        boot_strap = 0.0    

                        if (not done):
                            _, _, boot_strap = self.local_model.step(np.expand_dims(observation, axis=0), keep_p=self.current_prob)

                        discounted_bootstrapped_rewards = self.rewards_discounted(batch_rewards, self.gamma, boot_strap)
                        batch_rewards = discounted_bootstrapped_rewards
                        batch_advantages = np.array(batch_rewards) - np.array(batch_values)

                        all_values = np.concatenate((all_values, batch_values), 0) if all_values.size else np.array(batch_values)
                        all_advantages = np.concatenate((all_advantages, batch_advantages), 0) if all_advantages.size else np.array(batch_advantages)
                        all_rewards = np.concatenate((all_rewards, batch_rewards), 0) if all_rewards.size else np.array(batch_rewards)
                        all_states = np.concatenate((all_states, batch_states), 0) if all_states.size else np.array(batch_states)
                        all_actions = np.concatenate((all_actions, batch_actions), 0) if all_actions.size else np.array(batch_actions)

                           
                    # We can do this because: d/dx ∑ loss  == ∑ d/dx loss
                    data = (all_states, all_actions, all_rewards.tolist(), all_advantages, all_values)

                    if len(data[0]) != 0:
                        self.train(data) 
                    else:
                        break

                try:
                    #Save model and other variables
                    with open(save_path, 'w') as f:
                        try:
                            f.write(str(self.total_steps))
                            f.close()
                        except: 
                            raise
                    
                    self.global_model.save_model_weights()

                    print("Model saved")
                
                except:
                    print("ERROR: There was an issue saving the model!")
                    raise

            print("Training session was succesfull.")
            return True 

        except:
            print("ERROR: The coordinator ran into an issue during training!")
            raise
      
        self.plot.join()
        


