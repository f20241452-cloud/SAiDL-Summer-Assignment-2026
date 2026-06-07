import torch
import torch.nn as nn

class mLSTMCell(nn.Module):
    """Matrix memory LSTM cell with covariance update (Beck et al. 2024)."""
    def __init__(self, input_size: int, hidden_size: int, head_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        
        # Projections
        self.W_q = nn.Linear(input_size, head_dim)
        self.W_k = nn.Linear(input_size, head_dim)
        self.W_v = nn.Linear(input_size, head_dim)
        
        # Gates
        self.W_i = nn.Linear(input_size, 1)
        self.W_f = nn.Linear(input_size, 1)
        self.W_o = nn.Linear(input_size, hidden_size)

    def forward(self, x, C_prev, n_prev):
        """
        C_prev: Matrix memory state (batch, head_dim, head_dim)
        n_prev: Normalization state (batch, head_dim)
        """
        q_t = self.W_q(x)  # (B, d)
        k_t = self.W_k(x)  # (B, d)
        v_t = self.W_v(x)  # (B, d)
        
        i_t = torch.exp(self.W_i(x)) # Exponential gating
        f_t = torch.sigmoid(self.W_f(x)) 
        o_t = torch.sigmoid(self.W_o(x))
        
        # Matrix memory covariance update: C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)
        v_k_T = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1)) # (B, d, d)
        C_t = f_t.unsqueeze(-1) * C_prev + i_t.unsqueeze(-1) * v_k_T
        
        # Normalizer update
        n_t = f_t * n_prev + i_t * k_t
        
        # Retrieval
        h_tilde_t = torch.bmm(C_t, q_t.unsqueeze(2)).squeeze(2)
        h_tilde_t = h_tilde_t / (torch.sum(n_t * q_t, dim=1, keepdim=True) + 1e-8)
        
        h_t = o_t * h_tilde_t
        return h_t, C_t, n_t

class xLSTMPolicy(nn.Module):
    """Combined sLSTM and mLSTM backbone."""
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.embed = nn.Linear(state_dim, hidden_size)
        
        # Parallel pathways
        self.slstm = AdaptiveLSTMCell(hidden_size, hidden_size) # From your previous code
        self.mlstm = mLSTMCell(hidden_size, hidden_size, head_dim=hidden_size)
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        B, L, D = state_seq.shape
        x = self.embed(state_seq)
        
        # Initialize states
        h_s, c_s = torch.zeros(B, D, device=x.device), torch.zeros(B, D, device=x.device)
        C_m = torch.zeros(B, D, D, device=x.device)
        n_m = torch.zeros(B, D, device=x.device)
        
        # Process sequence
        for t in range(L):
            h_s, c_s = self.slstm(x[:, t, :], h_s, c_s)
            h_m, C_m, n_m = self.mlstm(x[:, t, :], C_m, n_m)
            
        # Combine representations at final timestep
        combined_h = torch.cat([h_s, h_m], dim=1)
        return self.action_head(combined_h)