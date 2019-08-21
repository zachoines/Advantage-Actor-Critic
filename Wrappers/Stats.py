# Python core libs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from collections import OrderedDict
from multiprocessing import Process, Lock
import threading
import time

# OpenAI Gym classes
import gym
from gym.core import RewardWrapper

# An action Wrapper class for environments
class Stats(RewardWrapper):
    def __init__(self, env, collector):
        super(Stats, self).__init__(env)
        self.episode_steps = 0
        self.numSteps = 0
        self.collector = collector
        self.CMA = 0
        self.SMA = lambda: self._updated_moving_average()
        self.last_values = 0
        self.EMA = 0
        self._recent_rewards = []
        self._buffer_len = 128
        self.TOTAL_EPISODE_REWARDS = 0
    def _update_buffer(self, reward):
        if len(self._recent_rewards) >= self._buffer_len:
            self._recent_rewards.pop(0)
            self._recent_rewards.append(reward)
        else:
            self._recent_rewards.append(reward)
    def _updated_moving_average(self):
        total_reward = 0
        for elem in self._recent_rewards:
            total_reward += elem
        return total_reward / self._buffer_len

    def reset(self, **kwargs):
        if self.numSteps > 0:
            self.collector.collect('CMA', self.CMA)
            self.collector.collect('LENGTH', self.episode_steps)
            self.collector.collect('TOTAL_EPISODE_REWARDS', self.TOTAL_EPISODE_REWARDS)
            self.collector.collect('EMA', self.EMA)
            self.collector.collect('SMA', self._updated_moving_average())

        self.episode_steps = 0
        self.TOTAL_EPISODE_REWARDS = 0
        
        # Dont reset EMA, CMA, SMA, or numSteps. These are online numbers.

        return self.env.reset(**kwargs)

    def step(self, action):
        self.numSteps += 1
        self.episode_steps += 1
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        self.TOTAL_EPISODE_REWARDS += reward

        # Cumulative moving average
        self.CMA = (reward + ((self.numSteps - 1) * self.CMA )) / self.numSteps

        # Exponential Moving Average (with 32 step smoothing)
        self.EMA = (reward * (2 / (1 + 32))) + (self.EMA * (1 - (2 / (1 + 32))))

        self._update_buffer(reward)
        return reward

class AsynchronousPlot(threading.Thread):
    def __init__(self, collector, live = False):
        super(AsynchronousPlot, self).__init__()
        self._live = live
        self._lines = []
        self._axis = []
        self.collector = collector
        self._busy = threading.Event()
        self._stoprequest = threading.Event()
        self._kill = threading.Event()
         
    def run(self):

        if self._live:
            self._fig = plt.figure(num = 0, figsize = (10, 6), dpi = 100)

            dims = self.collector.get_dimensions()
            
            if (len(dims) > 0):
                
                counter = 1
                for name in dims:

                    axis = self._fig.add_subplot(2,1, counter)
                    axis.autoscale_view()
                    line, = axis.plot([], [], label = name)
                    axis.set_xlabel('time')
                    axis.set_ylabel(name)
                    axis.add_line(line)
                    self._lines.append(line)
                    self._axis.append(axis)
                    counter += 1

            self._ani = animation.FuncAnimation(self._fig, self._update_graph, 
                interval = 50, 
                blit = True, 
                save_count = 100,
                repeat=True)

            plt.show()
        
        else:
            while not self._kill.isSet():
                self._write_results()
                time.sleep(5)
    def busy_notice(self):
        return self._busy.isSet()
    
    def continue_request(self):
        self._stoprequest.clear()
       
    def stop_request(self):
        self._stoprequest.set() 
        
    def join(self, timeout=None):
        
        if self._live:
            self._ani.save('Results.png', writer = "imagemagick", dpi = 80)
        
        # Signal the end of internal processes
        self._stoprequest.set()
        self._kill.set()

        super(AsynchronousPlot, self).join(timeout)

    def _draw_final_graph(self):
        pass

    def _write_results(self):

        if not self._stoprequest.isSet():

            self._busy.set()
            
            if (len(self.collector.dimensions) > 0):
                
                try:
    
                    for name in self.collector.dimensions:
                        
                        with open(".\stats\\" + name + ".txt", 'a') as f:
                            data = self.collector.get_dimension(name)

                            if  data == None or len(data) == 0:
                                continue

                            while len(data) > 0:
                                (x, (t, y)) = data.popitem(last = True)
                                f.write(str(t) + ", " + str(x) + ", " + str(y) + "\n")  

                except:
                    
                    f.close()
                    print("Issue writing results.")
                    raise

            self._busy.clear()
                

    def _update_graph(self, i):
        
        if not self._stoprequest.isSet():

            self._busy.set()
            
            if (len(self.collector.dimensions) > 0):
                
                for name in self.collector.dimensions:

                    data = self.collector.pop(name)

                    if data == None or len(data) == 0:
                        continue

                    (x, (t, y)) = data

                    for i in range(len(self._lines)):
                        line = self._lines[i]

                        label = line.get_label()
                        
                        if label == name:
                            
                            (xs, ys) = line.get_data()
                            xs = np.append(xs, [x], 0)
                            ys = np.append(ys, [y], 0)

                            line.set_xdata(xs)
                            line.set_ydata(ys)
                            
                            self._axis[i].set_xlim(min(xs), max(xs))
                            self._axis[i].set_ylim(min(ys), max(ys))
                            
                            self._busy.clear()
                            return self._lines

                self._busy.clear()
                return self._lines

            else:

                self._busy.clear()

                # Not ready to show anything
                return self._lines 

        else: 
            self._busy.clear()
            return self._lines
        

# Data structor for collecting multiple 2d dimentions
class Collector:
    def __init__(self):
        # datastructure is this: [( Data_name : { x_0 : y_0, x_1 : y_1, ....}), ....]
        self.current_nano_time = lambda: int(round(time.time_ns()))
        self.tstart = self.current_nano_time()
        self.dimensions = []
        self._data = []
        self._init = False

        # Because of async nature of access to collector,
        # we store total keys, and store on key number.
        self._num_keys = 0

    def collect(self, name, data):
        
        run_time = self.current_run_time()

        if (name not in self.dimensions):
            self.dimensions.append(name)
            entry = (name, OrderedDict([run_time, (self.current_nano_time(), data)]))
            self._data.append(entry)
            self._num_keys += 1

        elif self.get_dimension(name) == None:
            entry = (name, OrderedDict([]))
            self._data.append(entry)
            self._num_keys += 1
        else:
            
            try:
                ( _ , dim_values) = next((value for key, value in enumerate(self._data) if value[0] == name), None)
                dim_values[run_time] = (self.current_nano_time(), data) 
                self._num_keys += 1
            except GeneratorExit:
                pass
    

    
    def pop(self, name):
        data = self.get_dimension(name)
        
        if data == None or len(data) == 0:
            return None
        
        return data.popitem(last = True)

    def set_dimensions(self, dims):
        self.dimensions = dims
                
    def get_dimensions(self):
        return self.dimensions

    def get_dimension(self, name):

        if self._data == []:
            for n in self.dimensions:
                entry = (n, OrderedDict([]))
                self._data.append(entry)
            return None

        dim_values = None
        try:
            ( _ , dim_values) = next((value for key, value in enumerate(self._data) if value[0] == name), None)
        except GeneratorExit:
            pass
            

        return dim_values
            
    def remove(self, name):
        if name in self.dimensions:
            try:
                index = next((key for key, value in enumerate(self._data) if value[0] == name), None)
                return self._data.pop(index)
            except GeneratorExit:
                pass
            
            return None

    def current_run_time(self):
        return (self.current_nano_time() - self.tstart)
  

        


        