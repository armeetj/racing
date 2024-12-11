import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from networks import ReplayBuffer, ActorNetwork, CriticNetwork, ValueNetwork


class Agent:
    def __init__(
        self,
        alpha=0.00028,
        beta=0.00031,
        input_dims=[8],
        env=None,
        gamma=0.992,
        num_actions=2,
        max_size=1000000,
        tau=0.0048,
        batch_size=256,
        reward_scale=2,
    ):
        self.gamma_ = gamma
        self.tau_ = tau
        self.batch_size_ = batch_size
        self.num_actions_ = num_actions

        self.memory_ = ReplayBuffer(max_size, input_dims, num_actions)
        self.actor_ = ActorNetwork(
            num_actions=num_actions, max_action_value=env.action_space.high
        )
        self.critic_1_ = CriticNetwork(num_actions=num_actions, name="critic1")
        self.critic_2_ = CriticNetwork(num_actions=num_actions, name="critic2")
        self.value_ = ValueNetwork()
        self.target_value_ = ValueNetwork(name="target_value")

        self.actor_.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1_.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2_.compile(optimizer=Adam(learning_rate=beta))
        self.value_.compile(optimizer=Adam(learning_rate=beta))
        self.target_value_.compile(optimizer=Adam(learning_rate=beta))

        self.scale_ = reward_scale
        self.update_network(tau=1)

    def select_action(self, observation):
        current_state = tf.convert_to_tensor([observation])
        actions, _ = self.actor_.get_action(current_state)

        return actions[0]

    def add_to_memory(self, state, action, reward, next_state, is_terminal):
        self.memory_.save(state, action, reward, next_state, is_terminal)

    def update_network(self, tau=None):
        if tau is None:
            tau = self.tau_

        weights = []
        targets = self.target_value_.weights
        for idx, weight in enumerate(self.value_.weights):
            weights.append(weight * tau + targets[idx] * (1 - tau))

        self.target_value_.set_weights(weights)

    def learn(self):
        if self.memory_.counter_ < self.batch_size_:
            return

        state, action, reward, next_state, is_terminal = self.memory_.get_batch(
            self.batch_size_
        )

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value_(states), 1)
            next_value = tf.squeeze(self.target_value_(next_states), 1)

            current_policy_actions, log_probs = self.actor_.get_action(states)
            log_probs = tf.squeeze(log_probs, 1)

            q1_new_policy = self.critic_1_(states, current_policy_actions)
            q2_new_policy = self.critic_2_(states, current_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)
            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(
            value_loss, self.value_.trainable_variables
        )
        self.value_.optimizer.apply_gradients(
            zip(value_network_gradient, self.value_.trainable_variables)
        )

        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor_.get_action(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1_(states, new_policy_actions)
            q2_new_policy = self.critic_2_(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            # Here, log_probs represents entropy.
            # By minimizing actor_loss, we maximize the log probabilities of the actions, encouraging exploration over exploitation
            actor_loss = log_probs - critic_value

            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(
            actor_loss, self.actor_.trainable_variables
        )
        self.actor_.optimizer.apply_gradients(
            zip(actor_network_gradient, self.actor_.trainable_variables)
        )

        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale_ * reward + self.gamma_ * next_value * (1 - is_terminal)
            q1_old_policy = tf.squeeze(self.critic_1_(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2_(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)

        critic_1_network_gradient = tape.gradient(
            critic_1_loss, self.critic_1_.trainable_variables
        )
        critic_2_network_gradient = tape.gradient(
            critic_2_loss, self.critic_2_.trainable_variables
        )

        self.critic_1_.optimizer.apply_gradients(
            zip(critic_1_network_gradient, self.critic_1_.trainable_variables)
        )
        self.critic_2_.optimizer.apply_gradients(
            zip(critic_2_network_gradient, self.critic_2_.trainable_variables)
        )

        self.update_network()

    def save(self, reward):
        self.actor_.save_model(reward)
