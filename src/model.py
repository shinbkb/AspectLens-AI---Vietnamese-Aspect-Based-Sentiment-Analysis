"""Single-layer multi-aspect classifier matching the Kaggle checkpoint."""
from torch import nn

class AspectSentimentSLP(nn.Module):
    def __init__(self, input_dim: int, n_aspects: int = 10, n_states: int = 4):
        super().__init__()
        self.n_aspects = n_aspects
        self.n_states = n_states
        self.linear = nn.Linear(input_dim, n_aspects * n_states)

    def forward(self, features):
        return self.linear(features).reshape(-1, self.n_aspects, self.n_states)
