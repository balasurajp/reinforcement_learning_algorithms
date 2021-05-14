import gym
from gym.spaces import Discrete, Box
from gym.wrappers import TransformObservation, RescaleAction, ClipAction
from rlcluster.helpers.gymtools import Simplex
from rlcluster.envs.tradingmarkets.portfoliosystem import PortfolioEnvironment

CUSTOM_ENVIRONMENTS = {'PAM': PortfolioEnvironment}
ACTION_DESCRIPTIONS = {'categorical':(0,1), 'beta':(0,1), 'gaussian':(-1,1), 'dirichlet':(0,1), 'tanh':(-1,1), 'softmax':(0,1)}

def get_env_space(env: gym.Env):
    dim_state = env.observation_space.shape
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
        dim_action = (num_actions,)
    elif type(env.action_space) == Box:
        dim_action = env.action_space.shape
    elif type(env.action_space) == Simplex:
        num_actions = env.action_space.n
        dim_action = (num_actions,)
    else:
        raise NotImplementedError()
    return dim_state, dim_action

def get_environment(env_id, env_mode):
    envname = env_id.split('_')[0]
    if envname in CUSTOM_ENVIRONMENTS.keys():
        env = CUSTOM_ENVIRONMENTS[envname](env_id, env_mode)
    else:
        env = gym.make(env_id)
    return env

def make_compatible_environment(env_id:str, env_mode:str ='train', output_layer:str = None):
    env = get_environment(env_id, env_mode)

    if output_layer in ['beta', 'gaussian', 'tanh'] and isinstance(env.action_space, Box):
        env = RescaleAction(env, ACTION_DESCRIPTIONS[output_layer][0], ACTION_DESCRIPTIONS[output_layer][1])
        env = ClipAction(env)

    if isinstance(env.action_space, Box):
        action_type = 'continuous'
        assert output_layer in [None, 'tanh', 'beta', 'gaussian'], f"Discrete Actions are invalid for {output_layer} output_layer"
    elif isinstance(env.action_space, Discrete):
        action_type = 'discrete'
        assert output_layer in [None, 'categorical'], f"Discrete Actions are invalid for {output_layer} output_layer"
    elif isinstance(env.action_space, Simplex):
        action_type = 'simplex'
        assert output_layer in ['dirichlet', 'softmax'], f"Simplex action are invalid for {output_layer} output_layer"
    else:
        raise NotImplementedError()

    if len(env.observation_space.shape)==1:
        state_type = 'mlp'
    elif len(env.observation_space.shape)==3:
        state_type = 'cnn'
    else:
        raise NotImplementedError("Only MLP and CNN based models are available")

    dim_states, dim_actions = get_env_space(env)
    return env, dim_states, dim_actions, state_type, action_type