import math, torch
import torch.nn as nn
from rlcluster.models.networks import MLP_NET, CNN_NET
from rlcluster.helpers.torchtools import init_weight, resolve_activate_function


# BaseClass for ValueNet
class BaseValueNet(nn.Module):
    def __init__(self, dim_state=None, dim_action=None, dim_filter=32, dim_hidden=64, hidden_activation='relu', model_type=None):
        super(BaseValueNet, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_filter = dim_filter
        self.dim_hidden = dim_hidden
        self.hidden_activation = resolve_activate_function(hidden_activation)
        self.model_type = model_type
    
    def get_action(self, state):
        raise NotImplementedError()

# State Value Network - V(s) - (MLP,CNN)
class StateValueNetwork(BaseValueNet):
    def __init__(self, dim_state, dim_filter=32, dim_hidden=64, hidden_activation='relu', model_type=None):
        super(StateValueNetwork, self).__init__(dim_state, None, dim_filter, dim_hidden, hidden_activation, model_type)
        if model_type == 'mlp':
            self.model = MLP_NET(dim_state, (0,), (1,), dim_hidden, num_hidden_layers=1, hidden_activation=hidden_activation, output_layer='linear')
        elif model_type == 'cnn':
            self.model = CNN_NET(dim_state, (0,), (1,), dim_filter, dim_hidden, num_cnn_layers=1, num_ffn_layers=0, hidden_activation=hidden_activation, output_layer='linear')
        else:
            raise NotImplementedError(f'Invalid model type = {model_type}')
        self.model.apply(init_weight)

    def forward(self, state):
        value = self.model(state)
        return value

# State Action Value Network - Q(s,a) - (MLP,CNN)
class StateActionValueNet(BaseValueNet):
    def __init__(self, dim_state, dim_action, dim_filter=32, dim_hidden=64, hidden_activation='relu', model_type=None):
        super(StateActionValueNet, self).__init__(dim_state, dim_action, dim_filter, dim_hidden, hidden_activation, model_type)
        if model_type == 'mlp':
            self.model = MLP_NET(dim_state, dim_action, (1,), dim_hidden, num_hidden_layers=1, hidden_activation=hidden_activation, output_layer='linear')
        elif model_type == 'cnn':
            self.model = CNN_NET(dim_state, dim_action, (1,), dim_filter, dim_hidden, num_cnn_layers=1, num_ffn_layers=0, hidden_activation=hidden_activation, output_layer='linear')
        else:
            raise NotImplementedError(f'Invalid model type = {model_type}')
        self.model.apply(init_weight)
        
    def forward(self, state, action):
        qvalue = self.model(state, action)
        return qvalue

""" ============================================================================================================================================================ """

# Deep QValue Network : Q(a/s) 
class VanillaDQN(BaseValueNet):
    def __init__(self, dim_state, dim_action, dim_filter=32, dim_hidden=64, hidden_activation='relu', model_type=None):
        super(VanillaDQN, self).__init__(dim_state, dim_action, dim_filter, dim_hidden, hidden_activation, model_type)
        if model_type == 'mlp':
            self.model = MLP_NET(dim_state, (0,), dim_action, dim_hidden, num_hidden_layers=1, hidden_activation=hidden_activation, output_layer='linear')
        elif model_type == 'cnn':
            self.model = CNN_NET(dim_state, (0,), dim_action, dim_filter, dim_hidden, num_cnn_layers=1, num_ffn_layers=0, hidden_activation=hidden_activation, output_layer='linear')
        else:
            raise NotImplementedError(f'Invalid model type = {model_type}')
        self.model.apply(init_weight)

    def forward(self, state):
        q_values = self.model(state)
        return q_values

    def get_action(self, state):
        q_values = self.forward(state)
        max_action = q_values.max(dim=1)[1] 
        return max_action