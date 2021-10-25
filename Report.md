# Technical Details

This report contains technical details on the approach used in this project.

## Implementation

The reinforcement learning agent used in this project is based on double deep Q-learning. 

### Q-Updates

In this approach, two action-value functions `Q(s,a)` and `Q'(s,a)` are used, where `s` is the state and `a` the action. The first of these two functions is updated according to the rule

`Q(s,a) <- (1 - α) * Q(s,a) + α * (r + γ * max_a'Q'(s',a'))` (1)

where `α` is the learning rate, `r` the reward when going from the current state `s` to the next state `s'` and `γ` the discount factor. More on the details of this update will be shown further below.

The second function is updated via a soft update according to

`Q'(s,a) <- (1 - τ) * Q'(s,a) + τ * Q(s,a)`

with the soft update rate `τ`. Note that this update is not performed every frame but only every `frames per update` frame.

### Network topology

Each of the two action-value functions is represented by a fully connected neural network consisting of 4 hidden layers with 64 neurons per layer, yielding the network architecture

`37 -> 64 -> 64 -> 64 -> 64 -> 4`

The hidden layers have the `rectified linear unit` (=relu) as the activation function, whereas the output layer is purely linear.

If a state `s` and action `a` is given, the value of the action value function `Q(s,a)` is given by forward-propagating the state `s` through the neural network and looking at the activation value of the output neuron `a`.

### Backpropagation

Convergence is reached in (1) if 

`Q(s,a) = (1 - α) * Q(s,a) + α * (r + γ * max_a'Q'(s',a'))` 

or 

`r + γ * max_a'Q'(s',a') - Q(s,a) = 0` (2)

This is achieved via a backpropagation with the `batch size=64` of the neural network underlying the action-value function `Q(s,a)` with the mean square error loss function on the left hand side of equation (2) and an Adam optimizer.

### Policy

The policy is an epsilon greedy policy. I.e., there is an epsilon value, which starts at `ε=1` and is decreased to `ε=0.01` with the `ε decay factor=0.9999`, which is applied at each learning step. An epsilon greedy policy means that with a probability `ε` a random action is chosen and otherwise the action is chosen which yields the largest `Q(s,a)`-value.

### Replay memory

Also, a replay memory is used, which can store 10000 elements, where the oldest elements are discared if the limit of the memory is reached.


## Hyperparameters

A summary of the hyperparameters used to solve the environment is given in the following:

- `α = 0.001`
- `γ = 0.99`
- `ε interval = [0.01, 1]`
- `ε decay factor = 0.9999`
- `batch size = 64`
- `loss = mse`
- `τ = 0.001`
- `frames per update = 4`

- `max replay memory size = 100000`

- `number of hidden layers = 3`
- `hidden neurons per hidden layer = 64`
- `activation function = relu`

## Solution

As explained in the [README.md](README.md), the environment is considered as solved, if the average score over 100 consecutive episodes is at least +13. A solution of the environment was achieved in 605 episodes, as shown by the following screenshot from the Jupyter notebook:

![Episodes_Number](https://user-images.githubusercontent.com/92691697/137644425-f4d5895c-1ca6-4cf5-8391-bd4a571fbce7.PNG)

The score, i.e. non-discounted cumulative reward, per episode over the training process is shown in the following screenshot:

![Score](https://user-images.githubusercontent.com/92691697/137644387-1fbfb623-07f2-47a5-8ee8-b63c29746261.PNG)

## Ideas for improvements

Although the environment has been solved by the present approach, there are several possible ways to make improvements. Such improvements will impact in how many episodes the average score of +13 mentioned above is reached. And they will also affect the maximum average score reachable if the training would continue indefinitely.

The suggested improvements are the following ones:
- Continued manual adjustment of the hyperparameters: A certain amount of manual hyperparameter tuning (including network topology) was invested in this project. However, the upper limit has not yet been reached here. Unfortunetly, the tweaking of the hyperparameters becomes the more time intensive, the more fine-tuned they are.
- Auto machine learning: The hyperparameters can also be tuned automatically by performing a grid search or even better a random search.
- Usage of convolutional layers: The state space contains pixels of a camera. Although the state space is very small compared to usual applications of convolutional layers, one could explore their usage in even this case.
- Extension of state space by past: By using time delay neural networks or recurrent layers, the state space could be extended by the past states.
- Prioritized replay memory: The replay memory used in this project is not prioritized such that there is an improvement option.
- Duelling DDQN: The DDQN could be made duelling by using both an advantage A(s,a) and state-value function V(s) from which the action-value function Q(s,a) can be computed.
- Distributional approaches: In this approach, every state, action pair (s,a) has only a single scalar value Q. Distributional approaches extend this by providing a distribution over multiple Q-values.
- Rainbow: The paper [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf) provides a combination of multiple improvements. The [Dopamine](https://github.com/google/dopamine) framework of Google is an implementation of this framework and could also be used to solve the present environment.
- Attention: Primarily used in natural language processing, attention layers could also be explored in this context of this project.
