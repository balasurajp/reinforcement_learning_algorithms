import torch, numpy as np
from torch.optim import Adam
from torch import nn
from rlcluster.agents.templates.offpolicy_algorithm import OffPolicyAgent
from rlcluster.models.Policy import PolicyNetwork
from rlcluster.models.Value import StateActionValueNet
from rlcluster.helpers.torchtools import device, FLOAT
from rlcluster.helpers.torchtools import get_flat_params, set_flat_params


class TD3(OffPolicyAgent):
    def __init__(self, envid, render=False, seed=0, algoparams=dict(), netparams=dict(), savepath='./results/', debug=False):
        super(TD3, self).__init__(envid, render, seed, f"TD3_{algoparams['output_layer']}", algoparams, netparams, savepath, debug)
        ''' Twin Delayed Deep Deterministic Policy Gradient Agent - TD3 '''

        self.polyak_coefficient         = algoparams['polyak_coefficient']
        self.action_noise_scale         = algoparams['action_noise_scale']
        self.target_action_noise_scale  = algoparams['target_action_noise_scale']
        self.target_action_noise_clip   = algoparams['target_action_noise_clip']
        self.actor_update_delay         = algoparams['actor_update_delay']
        self.setup_logistics()
        self.setup_models()

    def setup_models(self):
        """Initialize NN models and load checkpoints if present"""
        assert self.actiontype in ['continuous', 'simplex'], f'Invalid action in TD3 algorithm - {self.actiontype}'

        # Initialize actor and critic networks
        self.actor_model = PolicyNetwork(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.output_layer, self.modeltype).to(device)
        self.actor_target = PolicyNetwork(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.output_layer, self.modeltype).to(device)
        self.critic_model1 = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)
        self.critic_target1 = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)
        self.critic_model2 = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)
        self.critic_target2 = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)

        self.actor_target.load_state_dict(self.actor_model.state_dict())
        self.critic_target1.load_state_dict(self.critic_model1.state_dict())
        self.critic_target2.load_state_dict(self.critic_model2.state_dict())

        self.actor_optim = Adam(self.actor_model.parameters(), lr=self.actor_lrp)
        self.critic_optim1 = Adam(self.critic_model1.parameters(), lr=self.critic_lrq)
        self.critic_optim2 = Adam(self.critic_model2.parameters(), lr=self.critic_lrq)

        # logging model objects and optimizers 
        self.logger.add_objects(actor_model=self.actor_model, critic_model1=self.critic_model1, critic_model2=self.critic_model2, 
                                actor_target=self.actor_target, critic_target1=self.critic_target1, critic_target2=self.critic_target2,
                                actor_optim=self.actor_optim, critic_optim1=self.critic_optim1, critic_optim2=self.critic_optim2)
        self.logger.load_objects()
        print('ACTOR: ', self.actor_model) 
        print('CRITIC: ', self.critic_model1)

    def choose_action(self, state, noise=True):
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor_model(state)
        action = action.cpu().numpy().squeeze(axis=0)

        if noise:
            noise_sample = (self.action_noise_scale * np.random.randn(self.dim_action[-1]))
            if self.output_layer=='tanh':
                action += noise_sample
                action = np.clip(action, -1.0, 1.0)
            elif self.output_layer=='softmax':
                noise_sample = np.exp(noise_sample) / np.exp(noise_sample).sum()
                noise_sample -= noise_sample.mean()
                action += noise_sample
            else:
                raise NotImplementedError()
        return action, None

    def core_update(self):
        ''' Core-update for TwinDelayed DDPG in off-policy fashion '''
        batch_data = self.rbuffer.sample_batch(self.batch_size)
        states, actions, rewards, next_states, dones = batch_data['states'], batch_data['actions'], batch_data['rewards'].unsqueeze(dim=1), batch_data['next_states'], batch_data['dones'].unsqueeze(dim=1)
        info = {'actor_loss':list(), 'critic_loss':list()}

        """core-update for critic"""
        with torch.no_grad():
            target_action = self.actor_target(next_states)
            target_action_noise = torch.clamp(torch.randn_like(target_action) * self.target_action_noise_scale, -self.target_action_noise_clip, self.target_action_noise_clip)
            target_action = torch.clamp(target_action + target_action_noise, -1.0, 1.0)
            target_values = rewards + self.gamma * dones * torch.min(self.critic_target1(next_states, target_action), self.critic_target2(next_states, target_action))

        """core-update for value1 target"""
        critic_pred1 = self.critic_model1(states, actions)
        critic_loss_1 = nn.MSELoss()(target_values, critic_pred1)

        self.critic_optim1.zero_grad()
        critic_loss_1.backward()
        self.critic_optim1.step()

        """core-update for value2 target"""
        critic_pred2 = self.critic_model2(states, actions)
        critic_loss_2 = nn.MSELoss()(target_values, critic_pred2)

        self.critic_optim2.zero_grad()
        critic_loss_2.backward()
        self.critic_optim2.step()
        info['critic_loss'].append( (critic_loss_1.item() + critic_loss_2.item()) / 2.0 )

        if self.logger.num_timesteps % self.actor_update_delay == 0:
            """core-update for policy net in TD3"""
            actor_loss = - self.critic_model1(states, self.actor_model(states)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            info['actor_loss'].append( actor_loss.item() )

            """polyak updating target nets in TD3"""
            actor_model_flat_params = get_flat_params(self.actor_model)
            actor_target_flat_params = get_flat_params(self.actor_target)
            set_flat_params(self.actor_target, (1.0 - self.polyak_coefficient) * actor_target_flat_params + (self.polyak_coefficient) * actor_model_flat_params)

            critic_model1_flat_params = get_flat_params(self.critic_model1)
            critic_target1_flat_params = get_flat_params(self.critic_target1)
            set_flat_params(self.critic_target1, (1.0 - self.polyak_coefficient) * critic_target1_flat_params + (self.polyak_coefficient) * critic_model1_flat_params)

            critic_model2_flat_params = get_flat_params(self.critic_model2)
            critic_target2_flat_params = get_flat_params(self.critic_target2)
            set_flat_params(self.critic_target2, (1.0 - self.polyak_coefficient) * critic_target2_flat_params + (self.polyak_coefficient) * critic_model2_flat_params)
        return info
        