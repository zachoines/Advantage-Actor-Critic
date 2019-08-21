# PPO_AI
AI model utilizing the Proximal Policy Optimization algorithm in the common Actor Critic style network strategy. This branch in particular utilizes the atari 128 byte ram environments from OpenAI.

### Installation
Should be able to get working with pip package manager. Simply type "pip install -r requirements.txt"
from the main dir. 

Microsoft Visual Studio 2015 Build Tools is Needed for the NES_PY module that is used to run environments I utilize. Download from [here.](https://www.microsoft.com/en-us/download/details.aspx?id=48159)

Additionally, make sure to edit "Main.py" for the approproate number of environments to run. Every environment is
run in it's own thread. Command line args will be added in the future.

### Main Features
Three main parts

#### AC2 Model and Training Loop
"Main.py" fires off a number of parallel "Worker" processes, each running an environment, and each sending batches of data to train a global "ACNetwork." A "Coordinator" class manages the parallel enviromments, providing fresh copies of the global network to each worker before the start of their next batch. 

#### Monitoring
For each epoch, a "Monitoring" wrapper class saves videos of the each environment's agent's progress. These videos are saved in the "Videos" dir.

#### Live statistics 
While the training runs, a "AsynchronousPlot" class utilizes a Matplotlib graph to constantly, and asynchronously, receive data from a "Collector" class. By default, only average rewards are collected from each worker thread's epocs. However, new dimensions can be added to the collector anytime, and the graph will update accordingly.
