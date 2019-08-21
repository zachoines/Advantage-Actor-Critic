# install requirements with:
# pip install -r requirements.txt

# Utility python classes
import threading
from multiprocessing import Process
import os
import sys
import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib

from Wrappers.preprocess import FrameSkip


print("GPU Available: ", tf.test.is_gpu_available())


# Importing the packages for OpenAI and MARIO
import gym


# Locally defined classes
from Wrappers import preprocess
from Wrappers.Normalize import Normalize
from Wrappers.Monitor import Monitor
from Wrappers.Stats import Stats, Collector, AsynchronousPlot
from Worker import Worker, WorkerThread
from AC_Network import AC_Model
from Coordinator import Coordinator

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
if gpus:
  try:
    
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# Environments to run
env_1 = 'MsPacman-ram-v0'
env_6 = 'CartPole-v0'
env_5 = 'MsPacman-ram-v4'
env_2 = "Breakout-ram-v0"
env_3 = "Pong-ramDeterministic-v4"
env_4 = "MsPacman-ramDeterministic-v4"
# env_names = [env_4, env_4, env_5, env_5]
env_names = [env_6, env_6, env_6, env_6]
# env_names = [env_1, env_1, env_1, env_1, env_1, env_1]
# env_names = [env_4]

# Configuration
current_dir = os.getcwd()
model_save_path = current_dir + '.\Model'
video_save_path = current_dir + '.\Videos'
record = True

# Enviromental vars
num_envs = len(env_names)
batch_size = 32
batches_per_epoch = 1000000
num_epocs = 512 * 10
gamma = .99
learning_rate = 7e-4
anneling_steps = num_epocs 

# Make the super mario gym environments and apply wrappers
envs = []
collector = Collector()
collector.set_dimensions(["CMA", "EMA", "SMA", "LENGTH", "LOSS", 'TOTAL_EPISODE_REWARDS'])
plot = AsynchronousPlot(collector, live=False)

# Apply env wrappers
counter = 0
for env in env_names:
    counter += 1
    env = gym.make(env)
    # env = Normalize(env)
    # env = FrameSkip(env, 4)
    env = Stats(env, collector)
    envs.append(env)

NUM_STATE = envs[0].observation_space.shape
NUM_ACTIONS = envs[0].env.action_space.n
ACTION_SPACE = envs[0].env.action_space

if not os.path.exists(video_save_path):
    os.makedirs(video_save_path)

if not os.path.exists('.\stats'):
    os.makedirs('.\stats')


workers = []
network_params = (NUM_STATE, batch_size, NUM_ACTIONS, ACTION_SPACE)

# Init Global and Local networks. Generate Weights for them as well.
Global_Model = AC_Model(NUM_STATE[0], NUM_ACTIONS, is_training=True, dropout_rate=0.5)
Global_Model(tf.convert_to_tensor(np.random.random((1, NUM_STATE[0]))))
step_model = AC_Model(NUM_STATE[0], NUM_ACTIONS, is_training=True, dropout_rate=0.5)
step_model(tf.convert_to_tensor(np.random.random((1, NUM_STATE[0]))))



# Load model if exists
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
else:
    try:
        if (os.path.exists(model_save_path + "\checkpoint")):
            
            Global_Model.load_model_weights()
            step_model.load_model_weights()

            for env in envs:
                workers.append(Worker(step_model, env, batch_size=batch_size, render=False))
            
            print("Model restored.")
        
        else:
            
            for env in envs:
                workers.append(Worker(step_model, env, batch_size=batch_size, render=False))
            
            print("Creating new model.")
    except:
        print("ERROR: There was an issue loading the model!")
        raise

coordinator = Coordinator(Global_Model, step_model, workers, plot, num_envs, num_epocs, batches_per_epoch, batch_size, gamma, model_save_path, anneling_steps)

# Train and save
if coordinator.run():
    try:
        Global_Model.save_model_weights()
        print("Model saved.")
        print("Now testing results....")
    except:
        print("ERROR: There was an issue saving the model!")
