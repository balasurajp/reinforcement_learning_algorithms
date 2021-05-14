import math, torch
import torch.nn as nn
from rlcluster.helpers.torchtools import init_weight, resolve_activate_function, resolve_output_function

# MLP Network
class MLP_NET(nn.Module):
    def __init__(self, dim_input, dim_branch, dim_output, dim_hidden=128, num_hidden_layers=1, hidden_activation='leakyrelu', output_layer='identity'):
        super(MLP_NET, self).__init__()
        self.dim_input = dim_input
        self.dim_branch = dim_branch
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_hidden_layers = num_hidden_layers
        assert num_hidden_layers >= 0, "Minimum hidden layers should be greater than zero"
        if dim_branch[0]!=0:
            infeatures = dim_input[0] + dim_branch[0]
        else:
            infeatures = dim_input[0]

        self.inputlayer = nn.Sequential(nn.Linear(infeatures, dim_hidden), resolve_activate_function(hidden_activation)()) 
        self.hiddenlayers = nn.Sequential()
        for idno in range(num_hidden_layers):
            self.hiddenlayers.add_module(f"HL{idno+1}", nn.Linear(dim_hidden, dim_hidden))
            self.hiddenlayers.add_module(f"HA{idno+1}", resolve_activate_function(hidden_activation)())
        self.outputlayer = resolve_output_function(output_layer)((dim_hidden,), dim_output)

    def forward(self, main_inputs, branch_inputs=None):
        if branch_inputs is not None and self.dim_branch[0]!=0:
            inputs = torch.cat([main_inputs, branch_inputs], dim=-1)
        else:
            inputs = main_inputs

        latent = self.inputlayer(inputs)
        latent = self.hiddenlayers(latent)
        outputs = self.outputlayer(latent)
        return outputs


# CNN Network
class CNN_NET(nn.Module):
    def __init__(self, dim_input, dim_branch, dim_output, dim_filter=64, dim_hidden=128, num_cnn_layers=1, num_ffn_layers=0, hidden_activation='leakyrelu', output_layer='linear'):
        super(CNN_NET, self).__init__()
        self.dim_input = dim_input
        self.dim_branch = dim_branch
        self.dim_output = dim_output
        self.dim_filter = dim_filter
        self.dim_hidden = dim_hidden
        self.num_cnn_layers = num_cnn_layers
        self.num_ffn_layers = num_ffn_layers
        
        assert (num_cnn_layers > 0) and (num_ffn_layers >= 0), "Minimum cnn and ffn layers should be greater than zero"
        kernalsize = (dim_input[2]+num_cnn_layers)//(num_cnn_layers+1)
        assert kernalsize > 1, "Too many CNN layers"
        kernaldelta = dim_input[2] - (num_cnn_layers+1)*kernalsize
        if kernaldelta>0:
            initial_kernal = (1, kernalsize+1)
            kernaldelta -= 1
        else:
            initial_kernal = (1, kernalsize)
        self.cnninputlayer = nn.Sequential(nn.Conv2d(dim_input[0], self.dim_filter, initial_kernal), resolve_activate_function(hidden_activation)())
        self.cnnhiddenlayers = nn.Sequential()
        for idno in range(num_cnn_layers):
            if kernaldelta>0:
                hidden_kernal = (1, kernalsize+1)
                kernaldelta -= 1
            else:
                hidden_kernal = (1, kernalsize)
            self.cnnhiddenlayers.add_module(f"CNN_HL{idno+1}", nn.Conv2d(self.dim_filter, self.dim_filter, hidden_kernal))
            self.cnnhiddenlayers.add_module(f"CNN_HA{idno+1}", resolve_activate_function(hidden_activation)())
        
        num_branch_features = dim_filter*dim_input[1] + dim_branch[0]
        self.ffninputlayer = nn.Sequential(nn.Linear(num_branch_features, self.dim_hidden), resolve_activate_function(hidden_activation)())
        self.ffnhiddenlayers = nn.Sequential()
        for idno in range(num_ffn_layers):
            self.ffnhiddenlayers.add_module(f"FFN_HL{idno+1}", nn.Linear(dim_hidden, dim_hidden))
            self.ffnhiddenlayers.add_module(f"FFN_HA{idno+1}", resolve_activate_function(hidden_activation)())
        self.outputlayer = resolve_output_function(output_layer)((dim_hidden,), dim_output)

    def forward(self, main_inputs, branch_inputs=None):
        latent = self.cnninputlayer(main_inputs)
        latent = self.cnnhiddenlayers(latent)
        
        latent = torch.flatten(latent, start_dim=1)
        if branch_inputs is not None and self.dim_branch[0] != 0 :
            latent = torch.cat([latent,branch_inputs], dim=-1)

        latent = self.ffninputlayer(latent)
        latent = self.ffnhiddenlayers(latent)

        outputs = self.outputlayer(latent)
        return outputs