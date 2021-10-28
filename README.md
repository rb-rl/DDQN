# DDQN
This repository contains a deep reinforcement learning agent based on a double deep Q-network (=DDQN) used for collecting food in a 3D Unity environment.

## Environment

The environment is a 3D room with bananas. It is based on the Unity engine and is provided by Udacity. The continuous states, discrete actions and the rewards are given as follows:

**State**

- 36 floating point values = pixels of camera attached to agent
- 1 floating point value = forward velocity of agent

**Action**

- 0 = move forward
- 1 = move backward
- 2 = turn left
- 3 = turn right

**Reward**

- +1 = agent collects yellow banana
- -1 = agent collects blue banana

The environment is episodic. The return per episode, which is the non-discounted cumulative reward, is referred to as a score. The environment is considered as solved if the score averaged over the 100 most recent episodes reaches +13.

## Demo

The repository adresses both training and inference of the agent. The training process can be observed in a Unity window, as shown in the following video.

https://user-images.githubusercontent.com/92691697/137642993-0aafe134-156c-4249-849b-0fd9a86432b3.mp4

When the training is stopped, the neural network of the agent is stored in a file called agent.pt.

The file [agent.pt](agent.pt) provided in this repository is the neural network of a successfully trained agent.

The application of the agent on the environment, i.e. the inference process, can also be observed in a Unity window with this repository:

https://user-images.githubusercontent.com/92691697/137643034-a6803ce3-b77e-4201-9198-20617e9b2b6b.mp4

## Installation

In order to install the project provided in this repository on Windows 10, follow these steps:

- Install a 64-bit version of [Anaconda](https://anaconda.cloud/installers)
- Open the Anaconda prompt and execute the following commands:
```
conda create --name drlnd python=3.6
activate drlnd

git clone https://github.com/udacity/deep-reinforcement-learning.git

cd deep-reinforcement-learning/python
```
- Remove `torch==0.4.0` in the file `requirements.txt` located in the current folder `.../python`
- Continue with the following commands:
```
pip install .
pip install keyboard
conda install pytorch=0.4.0 -c pytorch

python -m ipykernel install --user --name drlnd --display-name "drlnd"

cd ..\..

git clone git@github.com:rb-rl/DDQN.git
cd DDQN
```
- For Windows users: If you do not know whether you have a 64-bit operating system, you can use this [help](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
- Download the Udacity Unity Banana environment matching your environment:
  - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
  - [Amazon Web Services](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
- Unzip the zip file into the folder `DDQN` (for Windows (64-bit), the `Banana.exe` in the zip file should have the relative path `DDQN\Banana_Windows_x86_64\Banana.exe`, and for the other environments this path should be similar, but can also be adapted as shown further below)
- For Amazon Web Services users: You have to deactivate the [virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) and perform the training in headless mode. For inference, you have to activate the virtual screen and use the Linux Version above
- Start a jupyter notebook with the following command:
```
jupyter notebook
```
- Open `Main.ipynb`
- In the Jupyter notebook, select `Kernel -> Change Kernel -> drlnd`
- If you are not using Windows (64-bit), search for `UnityEnvironment("Banana_Windows_x86_64\Banana.exe")` in the notebook and update the path to the corresponding file of the environment you downloaded above

## Usage

In order to do the training and inference by yourself, simply open [Main.ipynb](Main.ipynb) and successively execute the Jupyter Notebook cells by pressing `Shift+Enter`.
