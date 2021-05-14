import math, torch
import torch.nn as nn
from torch.nn import functional as torchfn
from torch.distributions.kl import kl_divergence 
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.dirichlet import Dirichlet
from rlcluster.models.networks import MLP_NET, CNN_NET
from rlcluster.helpers.torchtools import init_weight, resolve_activate_function, resolve_output_function, clone_distribution

DETERMINISTIC_OUTPUTS = ['tanh', 'softmax', 'linear']
STOCHASTIC_OUTPUTS = ['gaussian', 'beta', 'dirichlet', 'categorical']

# Policy Network - (Stochastic, Deterministic) - (MLP, CNN)
class PolicyNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, dim_filter=32, dim_hidden=128, hidden_activation='relu', output_layer='gaussian', model_type='mlp'):
        super(PolicyNetwork, self).__init__()
        self.output_layer = output_layer
        if model_type == 'mlp':
            self.policy = MLP_NET(dim_state, (0,), dim_action, dim_hidden, num_hidden_layers=1, hidden_activation=hidden_activation, output_layer=output_layer)
        elif model_type == 'cnn':
            self.policy = CNN_NET(dim_state, (0,), dim_action, dim_filter, dim_hidden, num_cnn_layers=1, num_ffn_layers=0, hidden_activation=hidden_activation, output_layer=output_layer) 
        self.policy.apply(init_weight)

    def forward(self, state):
        output = self.policy(state)
        return output
    
    def get_log_prob_and_entropy(self, state, action):
        assert self.output_layer.lower() in STOCHASTIC_OUTPUTS, f'Invalid class method for Output = {self.output_layer}'
        distribution = self.forward(state)
        log_prob = distribution.log_prob(action)
        if len(log_prob.size())>1:
            log_prob = log_prob.sum(dim=-1)
        entropy = distribution.entropy()
        return log_prob, entropy

    def get_action_and_log_prob(self, state):
        assert self.output_layer.lower() in STOCHASTIC_OUTPUTS, f'Invalid class method for Output = {self.output_layer}'
        distribution = self.forward(state)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action)
        if len(log_prob.size())>1:
            log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def get_log_prob(self, state, action):
        assert self.output_layer.lower() in STOCHASTIC_OUTPUTS, f'Invalid class method for Output = {self.output_layer}'
        distribution = self.forward(state)
        log_prob = distribution.log_prob(action)
        if len(log_prob.size())>1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    def get_entropy(self, state):
        assert self.output_layer.lower() in STOCHASTIC_OUTPUTS, f'Invalid class method for Output = {self.output_layer}'
        distribution = self.forward(state)
        entropy = distribution.entropy()
        return entropy 

    def get_kldivergence(self, states):
        assert self.output_layer.lower() in STOCHASTIC_OUTPUTS, f'Invalid class method for Output = {self.output_layer}'
        distribution = self.forward(states)
        old_distribution = clone_distribution(distribution)
        kl = kl_divergence(old_distribution, distribution)
        return kl.sum(dim=1, keepdim=True)
        