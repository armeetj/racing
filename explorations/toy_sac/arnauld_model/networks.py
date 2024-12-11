import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense


class ReplayBuffer:
    def __init__(self, size, shape, n_actions):
        self.size_ = size  # Maximum size of the replay buffer
        self.counter_ = 0  # Number of experiences that have been added
        self.state_memory_ = np.zeros((self.size_, *shape))  # Memory array for states
        self.next_state_memory_ = np.zeros(
            (self.size_, *shape)
        )  # Memory array for the next state that was chosen
        self.action_memory_ = np.zeros(
            (self.size_, n_actions)
        )  # Memory array for the next action taken
        self.reward_memory_ = np.zeros(
            self.size_
        )  # Memory array for the action's reward
        self.terminal_memory_ = np.zeros(
            self.size_, dtype=bool
        )  # Memory array indicating whether a state is a terminal state

    def save(self, curr_state, action, reward, next_state, terminal):
        # Index describing where the new state should be stored
        idx = self.counter_ % self.size_

        # Store state information
        self.state_memory_[idx] = curr_state
        self.next_state_memory_[idx] = next_state
        self.action_memory_[idx] = action
        self.reward_memory_[idx] = reward
        self.terminal_memory_[idx] = terminal

        # Update the index
        self.counter_ += 1

    def get_batch(self, batch_size):
        # Randomly select batch_size experiences
        max_idx = min(self.counter_, self.size_)
        selected_idxs = np.random.choice(max_idx, batch_size)

        out_states = self.state_memory_[selected_idxs]
        out_actions = self.action_memory_[selected_idxs]
        out_rewards = self.reward_memory_[selected_idxs]
        out_next_states = self.next_state_memory_[selected_idxs]
        out_terminals = self.terminal_memory_[selected_idxs]

        return out_states, out_actions, out_rewards, out_next_states, out_terminals


class CriticNetwork(keras.Model):
    def __init__(self, num_actions, layer1_dim=256, layer2_dim=256, name="critic"):
        super(CriticNetwork, self).__init__()
        # Initialize network with 2 fully connected layers and 1 output layer
        self.layer1_dim_ = layer1_dim
        self.layer2_dim_ = layer2_dim
        self.num_actions_ = num_actions
        self.name_ = name

        self.layer1_ = Dense(self.layer1_dim_, activation="relu")
        self.layer2_ = Dense(self.layer2_dim_, activation="relu")
        self.q_ = Dense(1)

    def call(self, state, action):
        # Combine the state and action inputs into one input
        model_input = tf.concat([state, action], axis=1)

        # Pass the input through the fully connected layers to get q value
        q = self.layer1_(model_input)
        q = self.layer2_(q)
        q = self.q_(q)

        return q


class ValueNetwork(keras.Model):
    def __init__(self, layer1_dim=256, layer2_dim=256, name="value"):
        super(ValueNetwork, self).__init__()
        # Initialize network with 2 fully connected layers and 1 output layer
        self.layer1_dim_ = layer1_dim
        self.layer2_dim_ = layer2_dim
        self.name_ = name

        self.layer1_ = Dense(self.layer1_dim_, activation="relu")
        self.layer2_ = Dense(self.layer2_dim_, activation="relu")
        self.v_ = Dense(1)

    def call(self, state):
        # Pass the input through the fully connected layers to get v value
        v = self.layer1_(state)
        v = self.layer2_(v)
        v = self.v_(v)

        return v


class ActorNetwork(keras.Model):
    def __init__(
        self,
        max_action_value,
        layer1_dim=256,
        layer2_dim=256,
        num_actions=2,
        name="actor",
    ):
        super(ActorNetwork, self).__init__()
        # Initialize network with 2 fully connected layers, and two distinct output layers
        # mu layer used to calculate mean in action space
        # sigma layer used to calculate variance in action space
        self.layer1_dim_ = layer1_dim
        self.layer2_dim_ = layer2_dim
        self.num_actions_ = num_actions
        self.name_ = name
        self.max_action_value_ = max_action_value
        self.noise_ = 1e-6

        self.layer1_ = Dense(self.layer1_dim_, activation="relu")
        self.layer2_ = Dense(self.layer2_dim_, activation="relu")
        self.mu_ = Dense(self.num_actions_)
        self.sigma_ = Dense(self.num_actions_)

    def call(self, state):
        # Pass the state through the fully connected layers
        pred = self.layer1_(state)
        pred = self.layer2_(pred)

        # Gater the mean and variance describing the set of next possible action
        mean = self.mu_(pred)
        variance = self.sigma_(pred)
        variance = tf.clip_by_value(variance, self.noise_, 1)

        return mean, variance

    def get_action(self, state):
        # Gets the mean and variance of the set of next possible actions
        mean, variance = self.call(state)
        probabilities = tfp.distributions.Normal(mean, variance)

        # Randomly samples an action from the distribution
        possible_actions = probabilities.sample()
        action = tf.math.tanh(possible_actions) * self.max_action_value_

        log_probs = probabilities.log_prob(possible_actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise_)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

    # def get_config(self):
    #     base_config = super().get_config()
    #     config = {
    #         "sublayer": keras.saving.serialize_keras_object(self.sublayer),
    #     }
    #     return {**base_config, **config}

    # @classmethod
    # def from_config(cls, config):
    #     sublayer_config = config.pop("sublayer")
    #     sublayer = keras.saving.deserialize_keras_object(sublayer_config)
    #     return cls(sublayer, **config)
