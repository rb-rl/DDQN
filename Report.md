# Technical Details

This report contains technical details on the approach used in this project.

## Implementation

## Solution

As explained in the [Readme.md](Readme.md), the environment is considered as solved, if the average score over 100 consecutive episodes is at least +13. A solution of the environment was achieved in 605 episodes, as shown by the following screenshot from the Jupyter notebook:

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
