import torch
from torch.optim import Adam
from rlcluster.agents.templates.onpolicy_algorithm import OnPolicyAgent
from rlcluster.models.Policy import PolicyNetwork
from rlcluster.models.Value import StateValueNetwork
from rlcluster.helpers.torchtools import device, FLOAT

class VPG(OnPolicyAgent):
    def __init__(self, envid, render=False, seed=0, algoparams=dict(), netparams=dict(), savepath='./results/', debug=False):
        super(VPG, self).__init__(envid, render, seed, f"VPG_{algoparams['output_layer']}", algoparams, netparams, savepath, debug)
        '''  Vanilla Policy Gradient Agent - VPG '''

        self.actor_iterations   = algoparams['actor_iterations']
        self.critic_iterations  = algoparams['critic_iterations']
        self.max_gradnorm        = algoparams['max_gradnorm']
        self.l2_regularizer      = algoparams['l2_regularizer']
        self.setup_logistics()
        self.setup_models()

    def setup_models(self):
        """Initialize NN models and load checkpoints if present"""
        assert self.actiontype in ['continuous', 'discrete', 'simplex'], f'Invalid action type in VPG algorithm - {self.actiontype}'

        # Initialize actor and critic networks
        self.actor_model = PolicyNetwork(self.dim_state, self.dim_action, self.dim_filter, self.dim_hidden, self.hidden_activation, self.output_layer, self.modeltype).to(device)
        self.critic_model = StateValueNetwork(self.dim_state, self.dim_filter, self.dim_hidden, self.hidden_activation, self.modeltype).to(device)

        self.actor_optim = Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic_model.parameters(), lr=self.critic_lr)

        # logging model objects and optimizers 
        self.logger.add_objects(actor_model=self.actor_model, critic_model=self.critic_model, actor_optim=self.actor_optim, critic_optim=self.critic_optim)
        self.logger.load_objects()
        print('ACTOR: ', self.actor_model) 
        print('CRITIC: ', self.critic_model)

    def choose_action(self, state, noise=True):
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.actor_model.get_action_and_log_prob(state)
        action = action.cpu().numpy().squeeze(axis=0)
        log_prob = log_prob.cpu().numpy()
        return action, log_prob
    
    def compute_return(self, state):
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            value = self.critic_model(state)
        value = value.cpu().numpy().squeeze(axis=0)
        return value

    def core_update(self):
        """core-update for poilcy and value net in Vanilla Policy Gradient(VPG)"""
        batch_data = self.tbuffer.get_data()
        info = {'actor_loss':list(), 'critic_loss':list()}

        """update policy net in VPG"""
        for _ in range(self.actor_iterations):
            log_probs = self.actor_model.get_log_prob(batch_data['states'], batch_data['actions'])
            actor_loss = -(log_probs * batch_data['advantages']).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.max_gradnorm)
            self.actor_optim.step()
            info['actor_loss'].append( actor_loss.item() )

        """update value net in VPG"""
        for _ in range(self.critic_iterations):
            critic_pred = self.critic_model(batch_data['states'])

            critic_loss = torch.nn.MSELoss()(critic_pred, batch_data['returns'].unsqueeze(dim=1))
            for param in self.critic_model.parameters():
                critic_loss += param.pow(2).sum() * self.l2_regularizer
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            info['critic_loss'].append( critic_loss.item() )

        return info