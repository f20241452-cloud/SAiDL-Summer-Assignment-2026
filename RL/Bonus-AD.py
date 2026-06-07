class CrossEpisodeDistillationBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 500_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Flat arrays for continuous cross-episode history
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.returns_to_go = np.zeros((max_size, 1), dtype=np.float32)
        
        # Temporary storage for current episode
        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []

    def store_step(self, state, action, reward):
        self.ep_states.append(state)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)

    def finish_episode(self):
        """Calculates returns-to-go and commits the episode to the flat buffer."""
        ep_len = len(self.ep_rewards)
        rtg = np.zeros(ep_len, dtype=np.float32)
        
        discounted_sum = 0
        for i in reversed(range(ep_len)):
            discounted_sum = self.ep_rewards[i] + 0.99 * discounted_sum
            rtg[i] = discounted_sum
            
        for i in range(ep_len):
            self.states[self.ptr] = self.ep_states[i]
            self.actions[self.ptr] = self.ep_actions[i]
            self.returns_to_go[self.ptr] = rtg[i]
            
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            
        # Clear temporary buffers
        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []

    def sample_cross_episode_context(self, batch_size: int, seq_len: int = 32):
        """Samples sequences of length L, allowing crossing of episode boundaries."""
        # Ensure we don't sample wrapping around the circular buffer pointer
        valid_indices = np.arange(self.size - seq_len)
        if self.size == self.max_size:
            # Avoid the boundary where the pointer currently is writing
            invalid_start = max(0, self.ptr - seq_len)
            mask = (valid_indices < invalid_start) | (valid_indices > self.ptr)
            valid_indices = valid_indices[mask]
            
        start_idx = np.random.choice(valid_indices, batch_size)
        
        b_states = np.array([self.states[i : i + seq_len] for i in start_idx])
        b_actions = np.array([self.actions[i : i + seq_len] for i in start_idx])
        
        # The target for AD is to predict the action taken
        return (
            torch.FloatTensor(b_states),
            torch.FloatTensor(b_actions)
        )