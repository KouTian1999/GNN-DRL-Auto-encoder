import tensorflow as tf
import numpy as np


class DDPG():
    def __init__(self, env, buffer, actor, critic, actor_target, critic_target, gamma, tau, actor_lr, critic_lr, time_now):
        self.env = env
        self.buffer = buffer

        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.gamma = gamma
        self.tau = tau
        
        self.time_now = time_now

    def read_buffer(self):
        # state [batch_size, m_b, n_b]
        # action [batch_size, m_b, n_b]
        # reward [batch_size, 1]
        # next_state [batch_size, m_b, n_b]
        states, actions, rewards, next_states = self.buffer.get_batch()
        state_batch = tf.convert_to_tensor(states, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(actions, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_states, dtype=tf.float32)
        return state_batch, action_batch, reward_batch, next_state_batch

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = self.actor_target(next_state_batch, batch=True)
            y = reward_batch + self.gamma * self.critic_target(next_state_batch, target_actions, batch=True)
            critic_value = self.critic(state_batch, action_batch, batch=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, batch=True)
            critic_value = self.critic(state_batch, actions, batch=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        return critic_loss, actor_loss

    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def train(self, episode_num, step_num, reward_bound, noise=None):
        training = False
        avg_reward_list = []
        max_reward_list = []
        max_reward_all = 0
        for ep in range(episode_num):
            state = self.env.reset_state()
            episodic_reward = 0
            max_reward_ep = 0

            for i in range(step_num):
                tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
                action = self.actor(tf_state).numpy()

                if noise is not None:
                    noise_value = noise(tf.shape(action).numpy())
                    action = np.clip(action + noise_value, 0, 1)

                next_state, reward = self.env.reward(state, action)
                self.buffer.put_sample(state, action, reward, next_state)
                state = next_state

                episodic_reward += reward

                if reward > max_reward_ep:
                    max_reward_ep = reward
                    best_pcm_ep = next_state

                if reward > reward_bound and training is True:
                    np.savetxt('./Logs/' + self.time_now + '/mat/episode_' + str(ep) + 'step_' + str(i) + 'reward_' + str(reward) + '.txt', next_state, fmt='%s')
                    print('Matrix saved.')
                    self.actor.save_weights('./Logs/' + self.time_now + '/actor_model/episode_' + str(ep) + 'step_' + str(i) + 'reward_' + str(reward))
                    print('Actor model saved in ./actor_model/episode_' + str(ep) + 'step_' + str(i) + 'reward_' + str(reward))
                    self.critic.save_weights('./Logs/' + self.time_now + '/critic_model/episode_' + str(ep) + 'step_' + str(i) + 'reward_' + str(reward))
                    print('Critic model saved in ./critic_model/episode_' + str(ep) + 'step_' + str(i) + 'reward_' + str(reward))

                if self.buffer.current_size() >= self.buffer.batch_size:
                    training = True
                    state_batch, action_batch, reward_batch, next_state_batch = self.read_buffer()
                    critic_loss, actor_loss = self.update(state_batch, action_batch, reward_batch, next_state_batch)
                    self.update_target(self.actor_target.variables, self.actor.variables)
                    self.update_target(self.critic_target.variables, self.critic.variables)
                    print(f"Critic loss is ==> {critic_loss} * Actor loss is ==> {actor_loss}")

            avg_episodic_reward = episodic_reward/step_num
            avg_reward_list.append(avg_episodic_reward)
            max_reward_list.append(max_reward_ep)
            print(f"Episode * {ep} * Avg reward is ==> {avg_episodic_reward} * Max reward is ==> {max_reward_ep}")
            print(' ')

            if max_reward_ep > max_reward_all and training is True:
                max_reward_all = max_reward_ep
                best_pcm_all = best_pcm_ep

        print("The maximum reward we obtained during the whole training process is ", max_reward_all)

        return avg_reward_list, max_reward_list, best_pcm_all

    def test(self, step_num):
        state = self.env.reset_state()
        max_reward = 0
        for i in range(step_num):
            # without exploration noise
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
            action = self.actor(tf_state).numpy()
            next_state, reward = self.env.reward(state, action)
            state = next_state

            if reward > max_reward:
                max_reward = reward
                best_pcm = next_state

        return best_pcm, max_reward





