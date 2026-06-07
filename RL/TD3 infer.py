import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class Actor(nn.Module):

    def __init__(self, obs_size, act_size, action_limit):
        super().__init__()

        self.layer1 = nn.Linear(obs_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, act_size)

        self.action_limit = action_limit

    def forward(self, obs):

        obs = obs.to(device)

        hidden = F.relu(self.layer1(obs))
        hidden = F.relu(self.layer2(hidden))

        output = self.layer3(hidden)

        return self.action_limit * torch.tanh(output)


class Critic(nn.Module):

    def __init__(self, obs_size, act_size):
        super().__init__()

        inp = obs_size + act_size

        self.q1_l1 = nn.Linear(inp, 256)
        self.q1_l2 = nn.Linear(256, 256)
        self.q1_l3 = nn.Linear(256, 1)

        self.q2_l1 = nn.Linear(inp, 256)
        self.q2_l2 = nn.Linear(256, 256)
        self.q2_l3 = nn.Linear(256, 1)

    def forward(self, obs, act):

        combined = torch.cat(
            (obs, act),
            dim=1
        ).to(device)

        first = F.relu(self.q1_l1(combined))
        first = F.relu(self.q1_l2(first))
        first = self.q1_l3(first)

        second = F.relu(self.q2_l1(combined))
        second = F.relu(self.q2_l2(second))
        second = self.q2_l3(second)

        return first, second


class TD3Agent:

    def __init__(
        self,
        obs_size,
        act_size,
        action_limit,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4
    ):

        self.gamma = gamma
        self.tau = tau

        self.action_limit = action_limit

        self.actor = Actor(
            obs_size,
            act_size,
            action_limit
        ).to(device)

        self.actor_target = Actor(
            obs_size,
            act_size,
            action_limit
        ).to(device)

        self.critic = Critic(
            obs_size,
            act_size
        ).to(device)

        self.critic_target = Critic(
            obs_size,
            act_size
        ).to(device)

        self.actor_target.load_state_dict(
            self.actor.state_dict()
        )

        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )

        self.memory = []

        self.max_buffer_size = 2_000_000
        self.batch_size = 128

        self.delay = 2
        self.update_counter = 0

        self.noise_scale = 0.3

    def select_action(self, obs):

        obs_tensor = (
            torch.FloatTensor(obs)
            .unsqueeze(0)
            .to(device)
        )

        chosen_action = (
            self.actor(obs_tensor)
            .cpu()
            .data
            .numpy()
            .reshape(-1)
        )

        perturbation = np.random.normal(
            0,
            self.noise_scale * self.action_limit,
            size=chosen_action.shape
        )

        chosen_action = np.clip(
            chosen_action + perturbation,
            -self.action_limit,
            self.action_limit
        )

        self.noise_scale = max(
            0.1,
            self.noise_scale * 0.999
        )

        return chosen_action

    def store_transition(
        self,
        obs,
        act,
        reward,
        next_obs,
        done
    ):

        if len(self.memory) >= self.max_buffer_size:
            self.memory.pop(0)

        self.memory.append(
            (
                obs,
                act,
                reward,
                next_obs,
                float(done)
            )
        )

    def train(self):

        if len(self.memory) < self.batch_size:
            return

        sampled = random.sample(
            self.memory,
            self.batch_size
        )

        obs, acts, rewards, next_obs, dones = zip(*sampled)

        obs = torch.FloatTensor(obs).to(device)
        acts = torch.FloatTensor(acts).to(device)

        rewards = (
            torch.FloatTensor(rewards)
            .view(-1, 1)
            .to(device)
        )

        next_obs = (
            torch.FloatTensor(next_obs)
            .to(device)
        )

        dones = (
            torch.FloatTensor(dones)
            .view(-1, 1)
            .to(device)
        )

        future_actions = self.actor_target(
            next_obs
        ).clamp(
            -self.action_limit,
            self.action_limit
        )

        next_q1, next_q2 = self.critic_target(
            next_obs,
            future_actions
        )

        target_q = rewards + (
            (1 - dones)
            * self.gamma
            * torch.min(next_q1, next_q2).detach()
        )

        current_q1, current_q2 = self.critic(
            obs,
            acts
        )

        loss_q = (
            F.mse_loss(current_q1, target_q)
            +
            F.mse_loss(current_q2, target_q)
        )

        self.critic_optimizer.zero_grad()

        loss_q.backward()

        self.critic_optimizer.step()

        if self.update_counter % self.delay == 0:

            policy_loss = -self.critic(
                obs,
                self.actor(obs)
            )[0].mean()

            self.actor_optimizer.zero_grad()

            policy_loss.backward()

            self.actor_optimizer.step()

            self._soft_update(
                self.actor,
                self.actor_target
            )

            self._soft_update(
                self.critic,
                self.critic_target
            )

        self.update_counter += 1

    def _soft_update(
        self,
        source_net,
        target_net
    ):

        for tgt, src in zip(
            target_net.parameters(),
            source_net.parameters()
        ):

            tgt.data.copy_(
                self.tau * src.data
                +
                (1.0 - self.tau) * tgt.data
            )


env = gym.make(
    "Hopper-v5",
    render_mode="human",
    forward_reward_weight=10.0,
    ctrl_cost_weight=0.0069,
    healthy_reward=23.0,
    terminate_when_unhealthy=False
)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = TD3Agent(state_dim, action_dim, max_action)




agent.actor.load_state_dict(torch.load(________________________________________))
agent.actor.eval()


num_test_episodes = 10
total_rewards = []

for i in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    ep_reward = 0
    while not done:
        s_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
        action = agent.actor(s_tensor).cpu().data.numpy().flatten()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        ep_reward += reward
    total_rewards.append(ep_reward)
    print(f"Test Episode {i+1}: Reward = {ep_reward:.2f}")
env.close()
