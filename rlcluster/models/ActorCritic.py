import torch.nn as nn
from rlcluster.models.Policy import PolicyNetwork
from rlcluster.models.Value import StateValueNetwork


class ActorCriticNet(nn.Module):
    def __init__(self, dim_state, dim_action, dim_filter=32, dim_hidden=64, hidden_activation=nn.LeakyReLU, output_activation=None, action_type=None, model_type=None):
        super(ActorCriticNet, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_filter = dim_filter
        self.dim_hidden = dim_hidden
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.action_type = action_type
        self.model_type = model_type

        self.actor = PolicyNetwork(dim_state, dim_action, dim_filter, dim_hidden, hidden_activation, output_activation, action_type, model_type)
        self.critic = StateValueNetwork(dim_state, dim_filter, dim_hidden, hidden_activation, model_type)

    def forward(self, state):
        _, action, log_prob = self.actor.get_output(state)
        value = self.critic.forward(state)
        return action, log_prob, value

    def get_distribution(self, state):
        distribution, _, _ = self.actor.get_output(state)
        return distribution

    def get_action(self, state):
        _, action, _ = self.actor.get_output(state)
        return action

    def get_log_prob(self, state, action):
        distribution, _, _ = self.actor.get_output(state)
        log_prob = distribution.log_prob(action)
        return log_prob

    def get_entropy(self, state):
        distribution, _, _ = self.actor.get_output(state)
        entropy = distribution.entropy()
        return entropy 
    
    def get_value(self, state):
        value = self.critic.forward(state)
        return value