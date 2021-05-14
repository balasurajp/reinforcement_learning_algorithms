import numpy as np, pandas as pd, matplotlib.pyplot as plt
import gym, copy, os, pickle, json
from pprint import pprint
from gym.spaces import Box
from collections import deque
from rlcluster.helpers.gymtools import Simplex
from rlcluster.helpers.datatools import get_databasepath
from rlcluster.envs.tradingmarkets.utils import sharpe_ratio, max_drawdown, convert_decimal_to_binary
eps = 1e-8

class PortfolioSimulator(object):
    def __init__(self, lookback_window=3, num_assets=10, trading_commission=0.0025):
        self.lookback_window = lookback_window
        self.num_assets = num_assets
        self.trading_commission = trading_commission
        self.portfolio_returns = deque(maxlen=self.lookback_window)

    def calculate_portfolio_shrinkfactor(self, w0, w1):
        mu0 = 1
        mu1 = 1 - 2*self.trading_commission + self.trading_commission ** 2
        while abs(mu1-mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - self.trading_commission*w0[0] - (2*self.trading_commission - self.trading_commission**2) * np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / (1 - self.trading_commission*w1[0])
        mu1 = min(1.0, max(0.0, mu1))
        return mu1
    
    def get_information(self):
        pr = self.portfolio_returns.copy()
        pr = np.array(pr, dtype=np.float32)
        return pr

    def _reset(self):
        self.p0 = 1.0
        self.w0 = np.zeros(self.num_assets, dtype=np.float32)
        self.w0[0] = 1.0

        self.portfolio_returns.extend( [np.zeros(shape=(self.num_assets), dtype=np.float32) for _ in range(self.lookback_window)] )
        return self.get_information()

    def _step(self, w1, y1):
        assert self.w0.shape == w1.shape == y1.shape, 'w0, w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        w0 = self.w0.copy()
        mu = self.calculate_portfolio_shrinkfactor(w0, w1)
        p0 = self.p0
        p1 = mu * p0 * np.dot(y1, w1) 

        rate_of_return = p1 / p0 - 1
        immediate_reward = np.log((p1 + eps) / (p0 + eps))

        self.p0 = p1
        self.w0 = (y1*w1) / np.dot(y1, w1)
        done = (p1 == 0)

        self.portfolio_returns.append( (mu*y1*w1) - 1.0 )
        info = {"shrinkfactor": mu, "portfolio_value": p1, "rate_of_return": rate_of_return, "immediate_reward": immediate_reward, 
                "decision_weights": json.dumps(w1.tolist()), "evolved_weights": json.dumps(self.w0.tolist())}
        return self.get_information(), immediate_reward, done, info


class DataGenerator(object):
    def __init__(self, dataname='', lookback_window=7, episode_horizon=365, num_assets=15, mode='train'):
        self.lookback_window = lookback_window
        self.episode_horizon = episode_horizon
        self.num_assets = num_assets
        self.mode = mode
        self.load_dataset(dataname)

    def load_dataset(self, dataname):
        filepath =  os.path.join(get_databasepath(), 'assetmarket', f'{dataname}.pkl')
        with open(filepath, 'rb') as file:
            self.data, self.timestamps, self.asset_names = pickle.load(file)
        # keep data for required number of assets
        if len(self.asset_names)+1 > self.num_assets:
            self.asset_names = self.asset_names[:self.num_assets-1]
            self.data = self.data[:, :len(self.asset_names), :]
        # Split train and test data
        num_episodes = len(self.timestamps)//self.episode_horizon
        num_episodes_test = max(1, int(0.1*num_episodes))
        if self.mode == 'train':
            self.episodes = np.arange(len(self.timestamps))[self.lookback_window:-self.episode_horizon*(num_episodes_test+1)].tolist()
        else:
            self.episodes = [-self.episode_horizon*(i+1) for i in range(num_episodes_test)]

    def _reset(self):
        self.step = 0
        self.index = np.random.choice(self.episodes)
        self.episode_name = pd.Timestamp(self.timestamps[self.index])
        observation = self.data[self.index-self.lookback_window : self.index, :, :].copy()
        return observation

    def _step(self):
        self.step += 1
        self.index += 1

        info = {"timestamp":str(self.timestamps[self.index-1])}
        done = (self.step==self.episode_horizon)
        if done:
            next_observation = np.ones(shape=(self.lookback_window, self.num_assets-1, self.data.shape[-1]))
        else:
            next_observation = self.data[self.index-self.lookback_window : self.index, :, :].copy()
        return next_observation, done, info


class PortfolioEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, envid, mode='train'):
        self.decodeid(envid)
        self.mode = mode
        self.normalizeflag = False

        self.datarepo = DataGenerator(self.dataname, self.lookback_window, self.episode_horizon, self.num_assets, mode)
        self.simulator = PortfolioSimulator(self.lookback_window, self.num_assets, self.trading_commission)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.reset().shape, dtype=np.float32)
        self.action_space = Simplex(self.num_assets)
    
    def decodeid(self, envid):
        datasets = {'SD1': 'stocks1', 'SD2': 'stocks2', 'CD1': 'cryptos1'}
        _, dataname, num_assets, lookback_window, episode_horizon, state_type = envid.split("_")
        try:
            self.dataname = datasets[dataname]
            self.num_assets = int(num_assets.replace('A',''))
            self.lookback_window = int(lookback_window.replace('W',''))
            self.episode_horizon = int(episode_horizon.replace('H',''))
            self.state_type = int(state_type.replace('S',''))
            self.trading_commission = 0.0025
        except:
            print('Invalid identifiers for dataset')

    def transformAction(self, action):
        action = np.clip(action, 0, 1)
        weights = action
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)
        assert ((action >= 0) * (action <= 1)).all(), f'all action values should be between 0 and 1. Not {action}'
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3, err_msg=f'weights.sum() = 1 | weights={weights}')
        return weights
    
    def transformObservation(self, observation1, observation2):
        open_prices = observation1[:, :, 0]
        close_prices = observation1[:, :, 3]
        relative_pricechange = np.expand_dims((close_prices/open_prices)-1.0, 2)
        if self.state_type >= 1:
            observation = np.concatenate([np.zeros((self.lookback_window,1,1)), relative_pricechange], axis=1)
        if self.state_type >= 2:
            observation = np.concatenate([observation, np.expand_dims(observation2, 2)], axis=2)
        if self.state_type >= 3:
            stepinformation = np.arange(self.datarepo.step, self.datarepo.step+self.lookback_window, 1)
            stepencoding = np.array([convert_decimal_to_binary(step, int(np.ceil(np.log2(self.episode_horizon)))) for step in stepinformation], dtype=np.float32)
            stepreplication = np.array([stepencoding for _ in range(self.num_assets)], dtype=np.float32)
            observation3 = np.transpose(stepreplication, axes=(1,0,2))
            observation = np.concatenate([observation, observation3], axis=2)
        observation = np.transpose(observation, axes=(2,1,0))
        return observation

    def visualizeEpisode(self):
        infoframe = pd.DataFrame(self.infos)
        infoframe['timestamp'] = pd.to_datetime(infoframe['timestamp'])
        infoframe.set_index('timestamp', inplace=True)

        shr = sharpe_ratio(infoframe['rate_of_return'] + 1)
        mdd = max_drawdown(infoframe['rate_of_return'] + 1)
        title = f"{self.datarepo.episode_name} - (sharpe_ratio:{round(shr,1)}, max_drawdown:{round(mdd,1)})"
        infoframe[["portfolio_value"]].plot(title=title, fig=plt.gcf(), rot=30)
    
    def save_metadata(self, savepath):
        infoframe = pd.DataFrame(self.infos)
        infoframe['timestamp'] = pd.to_datetime(infoframe['timestamp'])
        infoframe.sort_values(['timestamp'], inplace=True)
        
        os.makedirs(savepath, exist_ok=True)
        filepath = f"{savepath}/{self.datarepo.episode_name}_{self.mode}.csv"
        infoframe.to_csv(filepath, index=False)

    def reset(self):
        observation1 = self.datarepo._reset()
        observation2 = self.simulator._reset()

        observation = self.transformObservation(observation1, observation2)
        self.infos = []
        return observation

    def step(self, action):
        np.testing.assert_almost_equal(action.shape, (self.num_assets,))
        np.testing.assert_almost_equal(np.sum(action), 1.0, 3, err_msg=f'action.sum() = {action.sum()} | action={action}')
        weights = self.transformAction(action)

        next_observation1, done1, info1 = self.datarepo._step()
        assets_price_evolution = np.concatenate([np.ones(1), next_observation1[-1, :, 3] / next_observation1[-1, :, 0]], axis=0)
        next_observation2, immediate_reward, done2, info2 = self.simulator._step(weights, assets_price_evolution)

        done, info = done1 or done2, {**info1, **info2}
        next_observation = self.transformObservation(next_observation1, next_observation2)
        self.infos.append(info)
        return next_observation, immediate_reward, done, info
            
    def render(self, mode='human'):
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.visualizeEpisode()
        else:
            print('nothing')

if __name__ == '__main__':
    print('Environment testing!')
    env = PortfolioEnvironment('PAM_SD1_A010_W003_H365_S001', 'test')
    print('num_episodes =', len(env.datarepo.episodes))
    buffer = []

    observation = env.reset()
    done = False
    while not done:
        action = np.ones((env.action_space.n,))
        action /= action.sum()

        # print(np.min(observation), np.max(observation), observation.shape)
        assert action.sum()-1.0 < 1e-3, f'Invalid action: {action}' 
        next_observation, reward, done, info = env.step(action)
        # print(env.datarepo.step)
        buffer.append( [observation, action, reward, next_observation, done, info] )
        observation = next_observation.copy()
        # env.render(mode='ansi')
    
    # env.saveMetadata('./results/test/')
    print("nsteps: ", len(buffer))