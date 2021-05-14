import torch, numpy as np
from torch.optim import Adam
from rlcluster.agents.templates.offpolicy_algorithm import OffPolicyAgent
from rlcluster.models.Policy import PolicyNetwork
from rlcluster.models.Value import StateActionValueNet
from rlcluster.helpers.torchtools import device, FLOAT
from rlcluster.helpers.torchtools import get_flat_params, set_flat_params


class SAC(OffPolicyAgent):
    def __init__(self, envid, render=False, seed=0, algoparams=dict(), netparams=dict(), savepath='./results/', debug=False):
        super(SAC, self).__init__(envid, render, seed, f"SAC_{algoparams['output_layer']}", algoparams, netparams, savepath, debug)
        ''' Soft Actor-Critic Agent - SAC '''

        self.alpha_lr            = algoparams['alpha_lr']
        self.polyak_coefficient  = algoparams['polyak_coefficient']
        self.setup_logistics()
        self.setup_models()

    def setup_models(self):
        """Initialize NN models and load checkpoints if present"""
        assert self.actiontype in ['continuous', 'simplex'], f'Invalid action in SAC algorithm - {self.actiontype}'
        self.target_entropy = - np.prod(self.trainenv.action_space.shape)

        # Initialize actor and critic networks
        self.alpha = torch.exp(torch.zeros(1, device=device)).requires_grad_()
        self.actor_model = PolicyNetwork(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.output_layer, self.modeltype).to(device)
        self.critic_modelq1 = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)
        self.critic_modelq2 = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)
        self.critic_targetq1 = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)
        self.critic_targetq2 = StateActionValueNet(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)

        self.critic_targetq1.load_state_dict(self.critic_modelq1.state_dict())
        self.critic_targetq2.load_state_dict(self.critic_modelq2.state_dict())

        self.alpha_optim = Adam([self.alpha], lr=self.alpha_lr)
        self.actor_optim = Adam(self.actor_model.parameters(), lr=self.actor_lrp)
        self.critic_optimq1 = Adam(self.critic_modelq1.parameters(), lr=self.critic_lrq)
        self.critic_optimq2 = Adam(self.critic_modelq2.parameters(), lr=self.critic_lrq)

        # logging model objects and optimizers 
        self.logger.add_objects(actor_model=self.actor_model, critic_modelq1=self.critic_modelq1, critic_modelq2=self.critic_modelq2, 
                                critic_targetq1=self.critic_targetq1, critic_targetq2=self.critic_targetq2, alpha=self.alpha, alpha_optim=self.alpha_optim, 
                                actor_optim=self.actor_optim, critic_optimq1=self.critic_optimq1, critic_optimq2=self.critic_optimq2)
        self.logger.load_objects()
        print('ACTOR: ', self.actor_model) 
        print('CRITIC: ', self.critic_modelq1)

    def choose_action(self, state, noise=True):
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.actor_model.get_action_and_log_prob(state)
        action = action.cpu().numpy().squeeze(axis=0)
        log_prob = log_prob.cpu().numpy()
        return action, log_prob

    def core_update(self):
        ''' Core-update for SAC in off-policy fashion '''
        batch_data = self.rbuffer.sample_batch(self.batch_size)
        states, actions, rewards, next_states, dones = batch_data['states'], batch_data['actions'], batch_data['rewards'].unsqueeze(dim=1), batch_data['next_states'], batch_data['dones'].unsqueeze(dim=1)
        info = {'actor_loss':list(), 'critic_loss':list()}

        """core-update for critic-q net"""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor_model.get_action_and_log_prob(next_states)
            q_target_values = torch.min(self.critic_targetq1(next_states, next_actions), self.critic_targetq2(next_states, next_actions))
            q_targets = rewards + self.gamma * dones * (q_target_values - self.alpha * next_log_probs.unsqueeze(dim=1))

        q_value_1 = self.critic_modelq1(states, actions)
        critic_loss_q1 = torch.nn.MSELoss()(q_value_1, q_targets)
        self.critic_optimq1.zero_grad()
        critic_loss_q1.backward()
        self.critic_optimq1.step()

        q_value_2 = self.critic_modelq2(states, actions)
        critic_loss_q2 = torch.nn.MSELoss()(q_value_2, q_targets)
        self.critic_optimq2.zero_grad()
        critic_loss_q2.backward()
        self.critic_optimq2.step()

        """core-update for actor-p net"""
        actions_pi, log_probs_pi = self.actor_model.get_action_and_log_prob(states)
        q1_values = self.critic_modelq1(states, actions_pi)
        q2_values = self.critic_modelq2(states, actions_pi)
        q_values = torch.min(q1_values, q2_values)

        actor_loss = (self.alpha * log_probs_pi.unsqueeze(dim=1) - q_values).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        """core-update for alpha"""
        alpha_loss = - self.alpha * (log_probs_pi.detach() + self.target_entropy).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        """core-update for target nets"""
        critic_targetq1_params = get_flat_params(self.critic_targetq1)
        critic_modelq1_params = get_flat_params(self.critic_modelq1)
        set_flat_params(self.critic_targetq1, (self.polyak_coefficient) * critic_modelq1_params + (1.0-self.polyak_coefficient) * critic_targetq1_params)
        
        critic_targetq2_params = get_flat_params(self.critic_targetq2)
        critic_modelq2_params = get_flat_params(self.critic_modelq2)
        set_flat_params(self.critic_targetq2, (self.polyak_coefficient) * critic_modelq2_params + (1.0-self.polyak_coefficient) * critic_targetq2_params)
        
        info['critic_loss'].append( (critic_loss_q1.item() + critic_loss_q2.item()) / 2.0 )
        info['actor_loss'].append( actor_loss.item() )
        return info