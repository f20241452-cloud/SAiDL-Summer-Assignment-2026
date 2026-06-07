import gymnasium as gym
import numpy as np

class HiddenVelocityWrapper(gym.ObservationWrapper):
    """Masks out the velocity components of the observation (Hopper-v5)."""
    def __init__(self, env):
        super().__init__(env)
        # In Hopper-v5, the first 5 dims are positional, the last 6 are velocities.
        self.pos_dim = 5
        
    def observation(self, obs):
        masked_obs = obs.copy()
        masked_obs[self.pos_dim:] = 0.0  # Zero out velocities to maintain tensor shapes
        return masked_obs

class GaussianNoiseWrapper(gym.ObservationWrapper):
    """Adds Gaussian noise to the observations."""
    def __init__(self, env, sigma=0.1):
        super().__init__(env)
        self.sigma = sigma
        
    def observation(self, obs):
        noise = np.random.normal(0, self.sigma, size=obs.shape)
        return (obs + noise).astype(np.float32)

class DelayedRewardWrapper(gym.Wrapper):
    """Accumulates rewards and yields them only every K steps."""
    def __init__(self, env, k=10):
        super().__init__(env)
        self.k = k
        self.step_count = 0
        self.accumulated_reward = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self.accumulated_reward += reward
        
        if self.step_count % self.k == 0 or terminated or truncated:
            yield_reward = self.accumulated_reward
            self.accumulated_reward = 0.0
        else:
            yield_reward = 0.0
            
        return obs, yield_reward, terminated, truncated, info

def make_pomdp_env(task_type="combined"):
    """Creates the environment based on the specific Bonus task."""
    env = gym.make('Hopper-v5')
    
    if task_type in ["hidden_vel", "combined"]:
        env = HiddenVelocityWrapper(env)
    if task_type == "combined":
        env = GaussianNoiseWrapper(env, sigma=0.1)
        env = DelayedRewardWrapper(env, k=10)
        
    return env