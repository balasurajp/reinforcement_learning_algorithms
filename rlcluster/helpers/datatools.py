import numpy as np, torch, pickle, logging
from os import path, makedirs
from os.path import abspath, join, dirname as parent
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


def get_databasepath():
    dirpath = parent(abspath(__file__)) # /helpers
    dirpath = parent(dirpath) # /rlcluster
    dirpath = parent(dirpath) # /rlalgorithms
    datapath = join(dirpath, 'database')
    return datapath


def get_resultspath():
    dirpath = parent(abspath(__file__)) # /helpers
    dirpath = parent(dirpath) # /rlcluster
    dirpath = parent(dirpath) # /rlalgorithms
    datapath = join(dirpath, 'results')
    return datapath


def get_envspath():
    dirpath = parent(abspath(__file__)) # /helpers
    dirpath = parent(dirpath) # /rlcluster
    datapath = join(dirpath, 'envs')
    return datapath


class RunningStatistics(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    def load_checkpoint(self, data):
        self._n = data[0]
        self._M = data[1]
        self._S = data[2]

    def get_checkpoint(self):
        data = (self._n, self._M, self._S)
        return data

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class NormalFilter:
    """
    Gaussian Normalisation :
    y = (x-mean)/std using running estimates of mean and std
    """

    def __init__(self, shape, normalize=True, clip=10.0):
        self.normalize = normalize
        self.clip = clip
        self.rs = RunningStatistics(shape)

    def __call__(self, x, update=True):
        if update and self.normalize:
            self.rs.push(x)

        if self.normalize:
            x = x - self.rs.mean
            x = x / (self.rs.std + 1e-8)
            if self.clip:
                x = np.clip(x, -self.clip, self.clip)
        return x
    
    def load_filter_state(self, data):
        self.rs.load_checkpoint(data)
    
    def get_filter_state(self):
        data = self.rs.get_checkpoint()
        return data

# Logs all information about agent
class agentlogger:
    def __init__(self, savepath, envname, agentname, debug, seedno) -> None:
        self.logpath     = path.join(savepath, 'logs', envname, agentname, f"seed{seedno}")
        self.modelpath   = path.join(savepath, 'models', envname, agentname, f"seed{seedno}")
        self.writer      = SummaryWriter(self.logpath)
        self.saveobjects = dict()
        self.traindata   = dict()
        self.timesteps   = 0
        self.debugflag   = debug

        self.torchobjectspath = path.join(self.modelpath, 'torchobjects.pth')
        self.otherobjectspath = path.join(self.modelpath, 'otherobjects.pkl')
        makedirs(self.logpath, exist_ok=True)
        makedirs(self.modelpath, exist_ok=True)
        self.create_message_logger()

    def create_message_logger(self):
        # Create a custom logger
        self.msglogger = logging.getLogger('rlcluster')
        if self.debugflag:
            self.msglogger.setLevel(logging.DEBUG)
        else:
            self.msglogger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(path.join(self.logpath, 'operations.log'), 'w')
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.msglogger.addHandler(c_handler)
        self.msglogger.addHandler(f_handler)

    def add_objects(self, **kwargs):
        for objectname, agentobject in kwargs.items():
            self.saveobjects[objectname] = agentobject
    
    def save_objects(self):
        otherdata = dict()
        torchdata = dict()
        for name, agentobject in self.saveobjects.items():
            if isinstance(agentobject, NormalFilter):
                otherdata[name] = agentobject.get_filter_state()
            elif isinstance(agentobject, (nn.Module, optim.Optimizer)):
                torchdata[name] = agentobject.state_dict()
            elif isinstance(agentobject, torch.Tensor):
                torchdata[name] = agentobject
            else:
                raise NotImplementedError(f"Object details: {name}, {type(agentobject)}")
        
        otherdata['timestep_counter'] = self.timesteps
        otherdata['train_data'] = self.traindata
        torch.save(torchdata, self.torchobjectspath)
        pickle.dump(otherdata, open(self.otherobjectspath, 'wb'))
    
    def load_objects(self):
        if path.exists(self.torchobjectspath) and path.exists(self.otherobjectspath):
            torchdata = torch.load(self.torchobjectspath)
            otherdata = pickle.load(open(self.otherobjectspath, 'rb'))

            self.timesteps = otherdata['timestep_counter']
            self.traindata = otherdata['train_data']
            for name, dobject in self.saveobjects.items():
                if isinstance(dobject, NormalFilter):
                    dobject.load_filter_state(otherdata[name])
                elif isinstance(dobject, (nn.Module, optim.Optimizer)):
                    dobject.load_state_dict(torchdata[name])
                elif isinstance(dobject, torch.Tensor):
                    dobject = torchdata[name]
                else:
                    raise NotImplementedError()
        else:
            self.msglogger.info('No checkpoints of saved objects are found!')

    def add_episodereward(self, episodereward):
        if 'episoderewards' not in self.traindata.keys():
            self.traindata['episoderewards'] = list()
        self.traindata['episoderewards'].append(episodereward)
    
    def add_traininginfo(self, updateinfo, updatetype):
        if updatetype == 'acm':
            if 'actor_loss' not in self.traindata.keys():
                self.traindata['actor_loss'] = list()
            if 'critic_loss' not in self.traindata.keys():
                self.traindata['critic_loss'] = list()            
            self.traindata['actor_loss'] += updateinfo['actor_loss']
            self.traindata['critic_loss'] += updateinfo['critic_loss']
        else:
            raise NotImplementedError()

    def summary_update(self, iterationNo, iterationtime, updatetype):
        avg_reward = np.around(np.mean(self.traindata['episoderewards'][-10:]), 3)
        num_episodes = len(self.traindata['episoderewards'])

        if updatetype == 'acm' and ('actor_loss' in self.traindata.keys()) and ('critic_loss' in self.traindata.keys()) :
            avg_actor_loss = np.mean(self.traindata['actor_loss'][-10:])
            avg_critic_loss = np.mean(self.traindata['critic_loss'][-10:])
            actor_updates = len(self.traindata['actor_loss'])
            critic_updates = len(self.traindata['critic_loss'])
            
            self.writer.add_scalar("timesteps/average_episode_reward", avg_reward, self.timesteps)
            self.writer.add_scalar("timesteps/actor_loss", avg_actor_loss, self.timesteps)
            self.writer.add_scalar("timesteps/critic_loss", avg_critic_loss, self.timesteps)
            self.writer.add_scalar("actor_updates/average_episode_reward", avg_reward, actor_updates)
            self.writer.add_scalar("actor_updates/actor_loss", avg_actor_loss, actor_updates)
            self.writer.add_scalar("actor_updates/num_timesteps", self.timesteps, actor_updates)
            self.writer.add_scalar("critic_updates/average_episode_reward", avg_reward, critic_updates)
            self.writer.add_scalar("critic_updates/critic_loss", avg_critic_loss, critic_updates)
            self.writer.add_scalar("critic_updates/num_timesteps", self.timesteps, critic_updates)

        self.msglogger.info(f"Iteration: {iterationNo} | Time: {iterationtime} | Timesteps:{self.timesteps} | Episodes:{num_episodes} | AverageEpisodeReward: {avg_reward}")
            
    def stepcounter(self):
        self.timesteps += 1 

    def debug(self, message=''):
        self.msglogger.debug(message)

    def warn(self, message=''):
        self.msglogger.warning(message)

    def info(self, message=''):
        self.msglogger.info(message)

    def error(self, message='Runtime Exception'):
        self.msglogger.error(message, exc_info=True)
    
    def critical(self, message=''):
        self.msglogger.critical(message)
    
    @property
    def num_timesteps(self):
        return self.timesteps

if __name__ == '__main__':
    dpath = get_databasepath()
    print(dpath)