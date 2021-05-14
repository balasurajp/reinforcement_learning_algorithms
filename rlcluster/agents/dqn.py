import torch, numpy as np
from torch.optim import Adam
from rlcluster.agents.templates.offpolicy_algorithm import OffPolicyAgent
from rlcluster.models.Value import VanillaDQN
from rlcluster.helpers.torchtools import device, FLOAT, LONG
from rlcluster.helpers.torchtools import get_flat_params, set_flat_params


class DQN(OffPolicyAgent):
    def __init__(self, envid, render=False, seed=0, algoparams=dict(), netparams=dict(), savepath='./results/', debug=False):
        super(DQN, self).__init__(envid, render, seed, f"DQN", algoparams, netparams, savepath, debug)
        ''' Deep Q-Network - DQN '''

        self.epsilon                = algoparams['epsilon']
        self.polyak_coefficient     = algoparams['polyak_coefficient']
        self.update_target_interval = algoparams['update_target_interval']
        self.setup_logistics()
        self.setup_models()

    def setup_models(self):
        """Initialize NN models and load checkpoints if present"""
        assert self.actiontype in ['discrete'], f'Invalid action in DQN algorithm - {self.actiontype}'

        # Initialize critic network
        self.value_model = VanillaDQN(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)
        self.value_target = VanillaDQN(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)

        self.value_target.load_state_dict(self.value_model.state_dict())
        self.value_optim = Adam(self.value_model.parameters(), lr=self.critic_lrq)

        # logging model objects and optimizers 
        self.logger.add_objects(critic_model=self.value_model, critic_target=self.value_target, critic_optim=self.value_optim)
        self.logger.load_objects()
        print('Q-NET: ', self.value_model)

    def choose_action(self, state, noise=True):
        state = FLOAT(state).unsqueeze(dim=0).to(device)
        if noise:
            if np.random.uniform() < self.epsilon:
                with torch.no_grad():
                    action = self.value_model.get_action(state)
                action = action.cpu().numpy().squeeze(axis=0)
            else:
                action = np.random.randint(0, self.dim_action[-1])
        else:
            with torch.no_grad():
                action = self.value_model.get_action(state)
            action = action.cpu().numpy().squeeze(axis=0)
        return action, None

    def core_update(self):
        ''' Core-update for DQN in off-policy fashion '''
        batch_data = self.rbuffer.sample_batch(self.batch_size)
        states, actions, rewards, next_states, dones = batch_data['states'], batch_data['actions'].long(), batch_data['rewards'].unsqueeze(dim=1), batch_data['next_states'], batch_data['dones'].unsqueeze(dim=1)
        info = {'actor_loss':list(), 'critic_loss':list()}
        
        """core-update for critic"""
        q_values = self.value_model(states).gather(1, actions)
        with torch.no_grad():
            q_target_next_values = self.value_model(next_states)
            q_target_actions = q_target_next_values.max(1)[1].view(q_values.size(0), 1)
            q_next_values = self.value_target(next_states)
            q_target_values = rewards + self.gamma * dones * q_next_values.gather(1, q_target_actions).view(q_values.size(0), 1)

        value_loss = torch.nn.MSELoss()(q_target_values, q_values)
        self.value_optim.zero_grad()
        value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 10.0)
        self.value_optim.step()        
        info['critic_loss'].append( value_loss.item() )

        if self.logger.num_timesteps % self.update_target_interval ==0:
            value_target_flat_params = get_flat_params(self.value_target)
            value_model_flat_params = get_flat_params(self.value_model)
            set_flat_params(self.value_target, (1.0-self.polyak_coefficient) * value_target_flat_params + self.polyak_coefficient * value_model_flat_params)

        """core-update for policy"""
        info['actor_loss'].append( np.array(0.0) )
        return info
