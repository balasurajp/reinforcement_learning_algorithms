import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.dirichlet import Dirichlet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOAT = torch.FloatTensor
DOUBLE = torch.DoubleTensor
LONG = torch.LongTensor


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Parameter):
        nn.init.constant_(m, 0.0)


def to_device(*args):
    return [arg.to(device) for arg in args]


def get_flat_params(model: nn.Module):
    """
    get tensor flatted parameters from model
    :param model:
    :return: tensor
    """
    return torch.cat([param.view(-1) for param in model.parameters()])


def get_flat_grad_params(model: nn.Module):
    """
    get flatted gradients of parameters from the model
    :param model:
    :return: tensor
    """
    return torch.cat( [param.grad.view(-1) if param.grad is not None else torch.zeros(param.view(-1).shape) for param in model.parameters()] )


def set_flat_params(model: nn.Module, flat_params: torch.Tensor):
    """
    set tensor flatted parameters to model
    :param model:
    :param flat_params: tensor
    :return:
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def resolve_activate_function(name: str):
    if name.lower() == "relu":
        activation = nn.ReLU
    if name.lower() == "sigmoid":
        activation = nn.Sigmoid
    if name.lower() == "leakyrelu":
        activation = nn.LeakyReLU
    if name.lower() == "prelu":
        activation = nn.PReLU
    if name.lower() == "softmax":
        activation = nn.Softmax
    if name.lower() == "tanh":
        activation = nn.Tanh
    if name.lower() == 'identity':
        activation = nn.Identity
    return activation

class StochasticGaussian(nn.Module):
    def __init__(self, dim_input, dim_output) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.paramlayer = nn.Linear(dim_input[0], 2*dim_output[0])

        self.min_log_std = -20
        self.max_log_std = 2
    
    def forward(self, latent):
        params = self.paramlayer(latent)
        mean, std = torch.split(params, self.dim_output[0], dim=-1)
        std = torch.clamp(std, self.min_log_std, self.max_log_std)
        std = torch.exp(std)
        dist = Normal(mean, std)
        return dist

class StochasticBeta(nn.Module):
    def __init__(self, dim_input, dim_output) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.paramlayer = nn.Linear(dim_input[0], 2*dim_output[0])

        self.min_param = -10
        self.max_param = 10
    
    def forward(self, latent):
        params = self.paramlayer(latent)
        params = torch.clamp(params, self.min_param, self.max_param)
        params = torch.nn.functional.softplus(params) + torch.ones_like(params)
        c1, c0 = torch.split(params, self.dim_output[0], dim=-1)
        dist = Beta(c1, c0)
        return dist

class StochasticDirichlet(nn.Module):
    def __init__(self, dim_input, dim_output) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.paramlayer = nn.Linear(dim_input[0], dim_output[0])

        self.min_param = -5
        self.max_param = 5
    
    def forward(self, latent):
        params = self.paramlayer(latent)
        params = torch.clamp(params, self.min_param, self.max_param)
        concentrations = torch.nn.functional.softplus(params) + torch.ones_like(params)
        dist = Dirichlet(concentrations)
        return dist

class StochasticCategorical(nn.Module):
    def __init__(self, dim_input, dim_output) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.paramlayer = nn.Linear(dim_input[0], dim_output[0])
    
    def forward(self, latent):
        params = self.paramlayer(latent)
        probs = torch.softmax(params, dim=-1)
        dist = Categorical(probs)
        return dist

class DeterministicTanh(nn.Module):
    def __init__(self, dim_input, dim_output) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.paramlayer = nn.Linear(dim_input[0], dim_output[0])
    
    def forward(self, latent):
        params = self.paramlayer(latent)
        output = torch.tanh(params)
        return output

class DeterministicSoftmax(nn.Module):
    def __init__(self, dim_input, dim_output) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.paramlayer = nn.Linear(dim_input[0], dim_output[0])
    
    def forward(self, latent):
        params = self.paramlayer(latent)
        output = torch.softmax(params, dim=-1)
        return output

class DeterministicLinear(nn.Module):
    def __init__(self, dim_input, dim_output) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.paramlayer = nn.Linear(dim_input[0], dim_output[0])
    
    def forward(self, latent):
        output = self.paramlayer(latent)
        return output

def resolve_output_function(name: str):
    if name.lower() == 'gaussian':
        output = StochasticGaussian
    elif name.lower() == 'beta':
        output = StochasticBeta
    elif name.lower() == 'dirichlet':
        output = StochasticDirichlet
    elif name.lower() == 'categorical':
        output = StochasticCategorical
    elif name.lower() == 'tanh':
        output = DeterministicTanh
    elif name.lower() == 'softmax':
        output = DeterministicSoftmax
    elif name.lower() == 'linear':
        output = DeterministicLinear
    else:
        raise NotImplementedError(f'Invalid output layer = {name}')
    return output

def clone_distribution(distribution):
    if isinstance(distribution, Normal):
        mean = torch.clone(distribution.mean).detach()
        stddev = torch.clone(distribution.stddev).detach()
        clonedist = Normal(mean, stddev)
    elif isinstance(distribution, Beta):
        concentration1 = torch.clone(distribution.concentration1).detach()
        concentration0 = torch.clone(distribution.concentration0).detach()
        clonedist = Beta(concentration1, concentration0)
    elif isinstance(distribution, Dirichlet):
        concentration = torch.clone(distribution.concentration).detach()
        clonedist = Dirichlet(concentration)
    elif isinstance(distribution, Categorical):
        concentration = torch.clone(distribution.probs).detach()
        clonedist = Categorical(concentration)
    return clonedist