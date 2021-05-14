import os, torch, time, numpy as np
from os import path
from rlcluster.helpers.memorytools import TrajectoriesBuffer
from rlcluster.helpers.envtools import make_compatible_environment
from rlcluster.helpers.datatools import NormalFilter, agentlogger


class OnPolicyAgent:
    def __init__(self, envid, render, seed, algoname, algoparams, netparams, savepath, debug):
        self.envid      = envid
        self.render     = render
        self.seed       = seed
        self.algoname   = algoname
        self.debugflag  = debug

        self.gamma      = algoparams['gamma']
        self.gaelambda  = algoparams['gaelambda']
        
        self.maxsteps_per_episode   = algoparams['maxsteps_per_episode']
        self.maxsteps_per_iteration = algoparams['maxsteps_per_iteration']
        
        self.actor_lr           = netparams['actor_lr']
        self.critic_lr          = netparams['critic_lr']
        self.hidden_activation  = netparams['hidden_activation']
        self.output_layer       = netparams['output_layer']
        self.dim_filter         = netparams['dim_filter']
        self.dim_hidden         = netparams['dim_hidden']

        self.savepath = savepath

    def setup_seed(self):
        """Initialize constant startseed for random generators"""
        if self.seed == 0:
            self.seed = np.random.randint(100000)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.trainenv.seed(self.seed)
        self.testenv.seed(self.seed)
    
    def setup_logistics(self):
        """Initialize environments and other supporting objects"""
        self.trainenv, self.dim_state, self.dim_action, self.modeltype, self.actiontype = make_compatible_environment(self.envid, 'train', self.output_layer)
        self.testenv, _, _, _, _ = make_compatible_environment(self.envid, 'test', self.output_layer)

        self.setup_seed()
        self.logger  = agentlogger(self.savepath, self.envid, self.algoname, self.debugflag, self.seed)
        self.logger.info(f"State dimension: {self.dim_state} | Action dimension: {self.dim_action}")
        try:
            nflag = self.trainenv.normalizeflag
        except:
            nflag = True

        self.statetransform = NormalFilter(shape=(self.dim_state[-1],), normalize=nflag)
        if nflag:
            self.logger.info(f'State Normalization is turned on! - {self.dim_state}')
        else:
            self.logger.warn(f'State Normalization is turned off! - {self.dim_state}')
        
        if self.actiontype == 'discrete':
            self.tbuffer = TrajectoriesBuffer(self.dim_state, (1,), self.maxsteps_per_iteration, self.gamma, self.gaelambda)
        else:
            self.tbuffer = TrajectoriesBuffer(self.dim_state, self.dim_action, self.maxsteps_per_iteration, self.gamma, self.gaelambda)
        
        self.logger.add_objects(state_transform=self.statetransform)

    def learn(self, num_iterations, num_evaluations, evaluate_interval, saving_interval):
        for num_iteration in range(num_iterations):
            num_iteration +=1
            start_time = time.time()
            # Agent-Environment Interaction
            while True:
                state = self.trainenv.reset()
                episode_reward = 0.0
                while True:
                    state = self.statetransform(state)
                    action, log_prob = self.choose_action(state)
                    value = self.compute_return(state)
                    next_state, reward, done, _ = self.trainenv.step(action)

                    self.tbuffer.store(state, action, log_prob, reward, value)
                    episode_reward += reward
                    self.logger.stepcounter()
                    state = next_state

                    episode_timeout  = (self.logger.num_timesteps % self.maxsteps_per_episode == 0)
                    iteration_timeout = (self.logger.num_timesteps % self.maxsteps_per_iteration == 0)
                    if done:
                        self.tbuffer.finish_trajectory(last_value=0.0)
                        break
                    if episode_timeout or iteration_timeout:
                        last_value = self.compute_return(next_state)
                        self.tbuffer.finish_trajectory(last_value=last_value)
                        break
                
                self.logger.add_episodereward(episode_reward)
                if iteration_timeout:
                    break
            
            # Agent learning
            updateinfo = self.core_update()
            self.logger.add_traininginfo(updateinfo, 'acm')

            # Agent Evaluation
            if num_iteration % evaluate_interval == 0:
                self.evaluate(num_evaluations)
            
            # Storing Agent
            if num_iteration % saving_interval == 0:
                self.logger.save_objects()  
            
            finish_time = time.time()
            iterationtime = round(finish_time-start_time, 2)
            self.logger.summary_update(num_iteration, iterationtime, 'acm')
            torch.cuda.empty_cache()

    def evaluate(self, num_evaluations):
        """Evaluate agent performance periodically and save metadata information"""
        metadatapath = path.join(self.savepath, 'metadata', self.envid, self.algoname, f"seed{self.seed}")
        os.makedirs(metadatapath, exist_ok=True)

        epsrewards = list()
        for evalno in range(num_evaluations):
            episode_reward = 0
            state = self.testenv.reset()

            while True:
                if self.render:
                    self.testenv.render()

                state = self.statetransform(state, update=False)
                action, _ = self.choose_action(state, noise=False)
                state, reward, done, _ = self.testenv.step(action)
                episode_reward += reward
                if done:
                    break

            epsrewards.append(episode_reward)
            try:
                self.testenv.save_metadata(metadatapath)
            except:
                self.logger.warn(f'No metadata for evaluation - EP{evalno}')

        print(f'Agent Evaluation ---> (mean, min, max) ---> ({np.mean(epsrewards)}, {np.min(epsrewards)}, {np.max(epsrewards)}) ')
        
    def setup_models(self):
        """Initialize NN models and load checkpoints if present"""
        raise NotImplementedError()

    def choose_action(self, state):
        ''' State ---> Action ---> Environment '''
        raise NotImplementedError()

    def compute_return(self, state):
        ''' Calculate Expected return from given state using current policy '''
        raise NotImplementedError()

    def core_update(self):
        ''' Core update of on-policy Algorithm '''
        raise NotImplementedError()
