import torch, numpy as np
from torch.optim import Adam
from rlcluster.agents.templates.offpolicy_algorithm import OffPolicyAgent
from rlcluster.models.Policy import PolicyNetwork
from rlcluster.models.Value import StateActionValueNet
from rlcluster.helpers.torchtools import device, FLOAT
from rlcluster.helpers.torchtools import get_flat_params, set_flat_params


class DDPG(OffPolicyAgent):
    def __init__(self, envid, render=False, seed=0, algoparams=dict(), netparams=dict(), savepath='./results/', debug=False):
        super(DDPG, self).__init__(envid, render, seed, f"DDPG_{algoparams['output_layer']}", algoparams, netparams, savepath, debug)
        ''' Deep Deterministic Policy Gradient - DDPG '''

        self.polyak_coefficient = algoparams['polyak_coefficient']
        self.action_noise_scale = algoparams['action_noise_scale']
        self.setup_logistics()
        self.setup_models()

    def setup_models(self):
        """Initialize NN models and load checkpoints if present"""
        assert self.actiontype in ['continuous', 'simplex'], f'Invalid action in DDPG algorithm - {self.actiontype}'

        # Initialize actor and critic networks
        self.actor_model = PolicyNetwork(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.output_layer, self.modeltype).to(device)
        self.critic_model = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)
        self.actor_target = PolicyNetwork(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.output_layer, self.modeltype).to(device)
        self.critic_target = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)

        self.actor_target.load_state_dict(self.actor_model.state_dict())
        self.critic_target.load_state_dict(self.critic_model.state_dict())

        self.actor_optim = Adam(self.actor_model.parameters(), lr=self.actor_lrp)
        self.critic_optim = Adam(self.critic_model.parameters(), lr=self.critic_lrq)

        # logging model objects and optimizers 
        self.logger.add_objects(actor_model=self.actor_model, critic_model=self.critic_model, actor_target=self.actor_target, critic_target=self.critic_target,
                                actor_optim=self.actor_optim, critic_optim=self.critic_optim)
        self.logger.load_objects()
        print('ACTOR: ', self.actor_model) 
        print('CRITIC: ', self.critic_model)

    def choose_action(self, state, noise=True):
        state = FLOAT(state).unsqueeze(dim=0).to(device)
        with torch.no_grad():
            action = self.actor_model(state)
        action = action.cpu().numpy().squeeze(axis=0)

        if noise:
            noise_sample = self.action_noise_scale * np.random.randn(self.dim_action[-1])
            if self.output_layer=='tanh':
                action += noise_sample
                action = np.clip(action, -1.0, 1.0)
            elif self.output_layer=='softmax':
                noise_sample = np.exp(noise_sample) / (np.exp(noise_sample).sum() + 1e-8)
                noise_sample -= noise_sample.mean()
                action += noise_sample
            else:
                raise NotImplementedError()
        return action, None

    def core_update(self):
        ''' Core-update for DDPG in off-policy fashion '''
        batch_data = self.rbuffer.sample_batch(self.batch_size)
        states, actions, rewards, next_states, dones = batch_data['states'], batch_data['actions'], batch_data['rewards'].unsqueeze(dim=1), batch_data['next_states'], batch_data['dones'].unsqueeze(dim=1)
        info = {'actor_loss':list(), 'critic_loss':list()}
        
        """core-update for critic"""
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_next_values = self.critic_target(next_states, next_actions)
            target_values = rewards + self.gamma * dones * target_next_values
        
        critic_pred = self.critic_model(states, actions)
        critic_loss = torch.nn.MSELoss()(critic_pred, target_values)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        info['critic_loss'].append( critic_loss.item() )

        """core-update for policy"""
        actions_pi = self.actor_model(states)
        actor_loss = -self.critic_model(states, actions_pi).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        """polyak update for target nets"""
        actor_model_flat_params = get_flat_params(self.actor_model)
        actor_target_flat_params = get_flat_params(self.actor_target)
        set_flat_params(self.actor_target, (1.0 - self.polyak_coefficient) * actor_target_flat_params + (self.polyak_coefficient) * actor_model_flat_params)

        critic_model_flat_params = get_flat_params(self.critic_model)
        critic_target_flat_params = get_flat_params(self.critic_target)
        set_flat_params(self.critic_target, (1.0 - self.polyak_coefficient) * critic_target_flat_params + (self.polyak_coefficient) * critic_model_flat_params)
        info['actor_loss'].append( actor_loss.item() )
        return info