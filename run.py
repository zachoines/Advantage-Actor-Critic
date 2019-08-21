# Utility python classes
import os
import sys
import numpy as np
import time
import tensorflow as tf

# Importing the packages for OpenAI and MARIO
import gym

# Locally defined classes
from AC_Network import AC_Model

# Environments to run
env = 'MsPacman-ram-v0'
env = 'CartPole-v0'
# env = "Breakout-ram-v0"

# Configuration
current_dir = os.getcwd()
model_save_path = current_dir + '.\Model'
record = True

# Enviromental vars
batch_size = 16

# Apply env wrappers
env = gym.make(env)
NUM_STATE = env.observation_space.shape
NUM_ACTIONS = env.env.action_space.n
ACTION_SPACE = env.env.action_space


network_params = (NUM_STATE, batch_size, NUM_ACTIONS, ACTION_SPACE)

# Init Global and Local networks. Generate Weights for them as well.
Test_Model = AC_Model(NUM_STATE, NUM_ACTIONS, is_training=False)
# Test_Model(tf.convert_to_tensor(np.random.random((1, 128))))

# Generate an batch worth of observations. Return nothing.
def run(num_steps, env, network, render):
    s = env.reset()
    done = False
    for step in range(num_steps):
        time.sleep(.00001)
        
        # Make a prediction and take a step if the epoc is not done
        if not done:
            [logits], d, _ = network.step(np.expand_dims(s, axis=0), 1.0)
            action = action_select(logits)
            s_t, reward, d, stuff = env.step(action)
            done = d
            s = s_t

            # render the env
            if (render and not done):
                env.render()
        
        else:
            done = False
            if (render):
                s = env.reset()
                env.render()
    print("Test run has finised.")
    return


# Action selection
def action_select(dist):
    dist = tf.nn.softmax(dist).numpy() 
    a = np.argmax(dist)
    return a

# Load model if exists
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
else:
    try:
        if (os.path.exists(model_save_path + "\checkpoint")):
            
            Test_Model.load_model_weights()
            print("Model restored.")
            print("Now running environment...")
            run(10000, env, Test_Model, True)
        
        else:
            raise("There is no model available")
    except:
        print("ERROR: There was an issue running the model!")
        raise


