import torch, math, time, numpy as np
from torch.optim import Adam
from rlcluster.agents.templates.onpolicy_algorithm import OnPolicyAgent
from rlcluster.models.Policy import PolicyNetwork
from rlcluster.models.Value import StateValueNetwork
from rlcluster.helpers.torchtools import device, FLOAT

class PPO(OnPolicyAgent):
    def __init__(self, envid, render=False, seed=0, algoparams=dict(), netparams=dict(), savepath='./results/', debug=False):
        super(PPO, self).__init__(envid, render, seed, f"PPO_{algoparams['output_layer']}", algoparams, netparams, savepath, debug)
        '''  Proximal Policy Optimization Agent - PPOClip '''
        self.entropy_coefficient = algoparams['entropy_coefficient']
        self.batch_size          = algoparams['batch_size']
        self.num_epochs          = algoparams['num_epochs']
        self.clip_epsilon        = algoparams['clip_epsilon']
        self.target_kldiv        = algoparams['target_kldiv']
        self.max_gradnorm        = algoparams['max_gradnorm']
        self.l2_regularizer      = algoparams['l2_regularizer']
        self.setup_logistics()
        self.setup_models()

    def setup_models(self):
        """Initialize NN models and load checkpoints if present"""
        assert self.actiontype in ['continuous', 'discrete', 'simplex'], f'Invalid action type in PPO algorithm - {self.actiontype}'

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
        """core-update for policy and value net in Proximal Policy Optimization(PPO)"""
        batch_data = self.tbuffer.get_data()
        states, actions, log_probs, returns, advantages = batch_data['states'], batch_data['actions'], batch_data['log_probs'], batch_data['returns'], batch_data['advantages'] 
        info = {'actor_loss':list(), 'critic_loss':list()}

        idx = np.random.permutation(len(states))
        for epochno in range(self.num_epochs):
            approx_kldivs = [] 
            for batchno in range(0, len(states), self.batch_size):
                """core-update for policy network in PPO"""
                rnidx = idx[batchno:batchno+self.batch_size] 
                mini_states, mini_actions, mini_log_probs, mini_returns, mini_advantages = states[rnidx], actions[rnidx], log_probs[rnidx], returns[rnidx], advantages[rnidx]

                log_probs_new, dist_entropy = self.actor_model.get_log_prob_and_entropy(mini_states, mini_actions)
                ratio = torch.exp(log_probs_new - mini_log_probs)

                policy_loss_1 = ratio * mini_advantages
                policy_loss_2 = torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * mini_advantages
                policy_loss  = torch.min(policy_loss_1, policy_loss_2).mean()
                entropy_loss = self.entropy_coefficient * dist_entropy.mean()
                actor_loss   = -(policy_loss + entropy_loss)

                approx_kldivs.append( (mini_log_probs - log_probs_new).mean().item() )
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.max_gradnorm)
                self.actor_optim.step()
                info['actor_loss'].append( actor_loss.item() )            

                """core-update for value network in PPO"""
                critic_pred = self.critic_model(mini_states)
                critic_loss = torch.nn.MSELoss()(critic_pred, mini_returns.unsqueeze(dim=1))
                for param in self.critic_model.parameters():
                    critic_loss += param.pow(2).sum() * self.l2_regularizer
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                info['critic_loss'].append( critic_loss.item() )

            if self.target_kldiv is not None and np.mean(approx_kldivs) > 1.5 * self.target_kldiv:
                self.logger.warn(f"KL divergence crossed permissible limits at epoch-{epochno} with approx. kldivergence-{np.mean(approx_kldivs)}")
                break
        return info