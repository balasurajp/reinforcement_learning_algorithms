import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy.lib.function_base import select
import gym, os, json
from copy import copy
from collections import deque
from pprint import pprint
from gym.spaces import Box
from collections import deque
from rlcluster.helpers.gymtools import Simplex
from rlcluster.helpers.datatools import get_databasepath
from rlcluster.envs.tradingmarkets.utils import sharpe_ratio, max_drawdown, convert_into_binary
eps = 1e-8


class PortfolioSimulator(object):
    def __init__(self, num_assets=0, trading_commission=0.0025):
        self.num_assets  = num_assets 
        self.commission  = trading_commission

    def calculate_portfolio_shrinkfactor(self, w0, w1):
        mu0 = 1
        mu1 = 1 - 2*self.commission + self.commission ** 2
        while abs(mu1-mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - self.commission*w0[0] - (2*self.commission - self.commission**2) * np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / (1 - self.commission*w1[0])
        mu1 = min(1.0, max(0.0, mu1))
        return mu1

    def _reset(self):
        self.p0 = 1.0
        self.w0 = np.zeros(self.num_assets, dtype=np.float32)
        self.w0[0] = 1.0

    def _step(self, w1, y1):
        assert self.w0.shape == w1.shape == y1.shape, 'w0, w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1, price change in cash component'

        mu = self.calculate_portfolio_shrinkfactor(self.w0.copy(), w1.copy())
        assert mu <= 1.0, f'Trading cost is larger than current holding value- {mu}'
        p0 = self.p0
        p1 = mu * p0 * np.dot(y1, w1)

        rate_of_return = (p1 / p0) - 1
        immediate_reward = np.log((p1 + eps) / (p0 + eps))

        self.p0 = p1
        self.w0 = (y1*w1) / np.dot(y1, w1)
        done = (p1 == 0)

        info = {"shrink_factor": mu, "portfolio_value": p1, "rate_of_return": rate_of_return, "immediate_reward": immediate_reward, 
                "portfolio_weights": json.dumps(w1.tolist()), "evoluted_weights": json.dumps(self.w0.tolist())}
        return immediate_reward, done, info


class DataGenerator(object):
    def __init__(self, dataname='', num_assets=10, lookback_window=7, episode_horizon=30, stateinfo=0, mode='train'):
        self.num_assets = num_assets
        self.lookback_window = lookback_window
        self.episode_horizon = episode_horizon
        self.stateinfo = stateinfo
        self.mode = mode
        filepath = os.path.join(get_databasepath(), 'assetmarket', f'{dataname}.ftr')

        self.dataframe = pd.read_feather(filepath)
        self.slice_data()
        self.slice_dataframe()

        self.lookback_porfolio_returns = deque(maxlen=lookback_window)
        self.lookback_episode_position = deque(maxlen=lookback_window)

    def slice_data(self):
        timestamps = self.dataframe.timestamp.unique()
        timestamps = sorted(timestamps)

        totalepisodes = len(timestamps)//self.episode_horizon
        ntestepisodes = max(1, int(0.1*totalepisodes))

        if self.mode=='test':
            episodes = []
            for epno in range( ntestepisodes ):
                idx = (epno+1)*self.episode_horizon
                episodes.append(timestamps[-idx])
        elif self.mode=='train':
            date_1 = timestamps[ self.lookback_window ]
            date_2 = timestamps[ -(ntestepisodes+1)*self.episode_horizon ]
            episodes = [timestamp for timestamp in timestamps if (timestamp>=date_1) and (timestamp<=date_2)]
        else:
            raise NotImplementedError()
        
        self.timedelta = pd.Timestamp(timestamps[1]) - pd.Timestamp(timestamps[0])
        self.episodes = episodes.copy()
    
    def slice_dataframe(self):
        miniframe = self.dataframe.loc[:, ['asset', 'close', 'volume']]
        miniframe['valuation'] = miniframe['close'] * miniframe['volume']
        assets = miniframe[['asset','valuation']].groupby('asset').std().sort_values('valuation', ascending=False).index.tolist()
        if len(assets) < self.num_assets:
            self.asset_names = assets
            self.num_assets = len(assets) + 1
        else:
            self.asset_names = assets[:self.num_assets-1]
        self.dataframe = self.dataframe.loc[self.dataframe.asset.isin(self.asset_names), :].reset_index()
        self.dataframe.set_index(['asset', 'timestamp'], inplace=True)
        self.dataframe.sort_values(['asset', 'timestamp'], inplace=True)

    def binary_length(self):
        return int(np.ceil( np.log2(self.episode_horizon) ))
    
    def get_observation(self, timestamp, portfolio_return=0.0, episode_position=None):
        time_1 = timestamp - self.lookback_window * self.timedelta
        time_2 = timestamp - 1 * self.timedelta

        # asset_relative_prices = (self.dataframe.loc[(self.asset_names, slice(time_1,time_2)), 'relative_close'].unstack().values) - 1.0
        open_prices = (self.dataframe.loc[(self.asset_names, slice(time_1,time_2)), 'open'].unstack().values)
        close_prices = (self.dataframe.loc[(self.asset_names, slice(time_1,time_2)), 'close'].unstack().values)
        asset_relative_prices = (close_prices/open_prices) - 1.0

        if episode_position == 0:
            self.lookback_porfolio_returns.clear()
            self.lookback_episode_position.clear()
            self.lookback_porfolio_returns.extend(np.zeros(self.lookback_window, dtype=np.float32))
            self.lookback_episode_position.extend(np.zeros(self.lookback_window, dtype=np.int32))
        self.lookback_porfolio_returns.append(portfolio_return)
        self.lookback_episode_position.append(episode_position)
        portfolio_returns = np.reshape(np.array(self.lookback_porfolio_returns.copy()), (1, self.lookback_window))
        episode_positions = np.reshape(convert_into_binary(self.lookback_episode_position, self.binary_length()), (self.binary_length(), self.lookback_window))
            
        if self.stateinfo == 1:
            data = np.concatenate([np.ones((1,self.lookback_window), dtype=np.float32), asset_relative_prices], axis=0)
        elif self.stateinfo == 2:
            data = np.concatenate([np.ones((1,self.lookback_window), dtype=np.float32), asset_relative_prices, portfolio_returns], axis=0)
        elif self.stateinfo == 3:
            data = np.concatenate([np.ones((1,self.lookback_window), dtype=np.float32), asset_relative_prices, portfolio_returns, episode_positions], axis=0)
        else:
            raise NotImplementedError()
        
        state = np.array(np.expand_dims(data, axis=2), dtype=np.float32)
        state = np.transpose(state, (2,1,0))
        return state

    def get_groundtruth(self, timestamp):
        if self.current_episode_timestep==self.episode_horizon:
            relative_prices = np.ones(shape=(self.num_assets,), dtype=np.float32)
        else:
            groundtruth_time = timestamp
            # asset_relative_prices = self.dataframe.loc[(self.asset_names, groundtruth_time), 'relative_close'].values
            open_prices = (self.dataframe.loc[(self.asset_names, groundtruth_time), 'open'].values)
            close_prices = (self.dataframe.loc[(self.asset_names, groundtruth_time), 'close'].values)
            asset_relative_prices = (close_prices/open_prices)
            relative_prices = np.concatenate([np.ones((1,), dtype=np.float32), asset_relative_prices], axis=0)
        return relative_prices

    def _reset(self):
        self.current_episode_nametag = pd.Timestamp(np.random.choice(self.episodes))
        self.current_episode_timestamp = copy(self.current_episode_nametag)
        self.current_episode_timestep = 0
        state = self.get_observation(self.current_episode_timestamp, 0.0, self.current_episode_timestep)
        groundtruth = self.get_groundtruth(self.current_episode_timestamp)
        return state, groundtruth

    def _step(self, portfolio_return):
        info = {'timestamp': str(self.current_episode_timestamp), 'portfolio_return': portfolio_return}

        self.current_episode_timestamp = self.current_episode_timestamp + self.timedelta
        self.current_episode_timestep += 1
        nextstate = self.get_observation(self.current_episode_timestamp, portfolio_return, self.current_episode_timestep)
        nextgroundtruth = self.get_groundtruth(self.current_episode_timestamp)
        
        done = True if self.current_episode_timestep==self.episode_horizon else False
        return nextstate, nextgroundtruth, done, info


class PortfolioEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, envid, mode='train'):
        self.decodeid(envid)
        self.normalizeflag = False
        self.mode = mode

        self.datasim = DataGenerator(self.dataname, self.num_assets, self.lookback_window, self.episode_horizon, self.state_type, mode)
        self.portsim = PortfolioSimulator(self.datasim.num_assets, self.trading_commission)

        self.action_space = Simplex(self.datasim.num_assets)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.reset().shape, dtype=np.float32)
    
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

    def reset(self):
        self.portsim._reset()
        state, self.groundtruth = self.datasim._reset()
        self.info = list()
        return state

    def step(self, action):
        np.testing.assert_almost_equal(np.sum(action), 1.0, 3, err_msg=f'portfolio_weights.sum() = {action.sum()} | portfolio_weights={action}')
        w1, y1 = action.copy(), self.groundtruth.copy()

        reward, done1, info1 = self.portsim._step(w1, y1)
        nextstate, nextgroundtruth, done2, info2 = self.datasim._step(info1['rate_of_return'])
        done, info = done1 or done2, {**info1, **info2}

        self.groundtruth = nextgroundtruth
        self.info.append(info)
        return nextstate, reward, done, info
            
    def render(self, mode='ansi'):
        if mode == 'ansi':
            pprint(self.info[-1])
        elif mode == 'human':
            self.render_plot()
        else:
            print('nothing')

    def render_plot(self):
        infoframe = pd.DataFrame(self.info)
        infoframe['timestamp'] = pd.to_datetime(infoframe['timestamp'])
        infoframe.set_index('timestamp', inplace=True)

        shr = sharpe_ratio(infoframe['rate_of_return'] + 1)
        mdd = max_drawdown(infoframe['rate_of_return'] + 1)
        title = f"{self.datasim.current_episode_nametag} - (sharpe_ratio:{round(shr,1)}, max_drawdown:{round(mdd,1)})"
        infoframe[["portfolio_value"]].plot(title=title, fig=plt.gcf(), rot=30)
    
    def save_metadata(self, savepath):
        infoframe = pd.DataFrame(self.info)
        infoframe['timestamp'] = pd.to_datetime(infoframe['timestamp'])
        infoframe.sort_values(['timestamp'], inplace=True)
        
        os.makedirs(savepath, exist_ok=True)
        filepath = f"{savepath}/{self.datasim.current_episode_nametag}_{self.mode}.csv"
        infoframe.to_csv(filepath, index=False)

if __name__ == '__main__':
    print('Environment testing!')
    env = PortfolioEnvironment('PAM_SD1_A010_W003_H365_S001', 'test')
    print(f'num_episodes = {len(env.datasim.episodes)}')
    buffer = []

    observation = env.reset()
    done = False
    while not done:
        action = np.ones((env.action_space.n,))
        action /= action.sum()

        # print(np.min(observation), np.max(observation), observation.shape)
        # env.render(mode='ansi')

        assert action.sum()-1.0 < 1e-3, f'Invalid action: {action}' 
        next_observation, reward, done, info = env.step(action)
        buffer.append( [observation, action, reward, next_observation, done, info] )
        observation = next_observation.copy()

    # env.saveMetadata('./results/test/')
    print("nsteps: ", len(buffer))