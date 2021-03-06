#!/usr/bin/env python
# coding: utf-8

# In[6]:


import datetime
import sys, os,inspect 
#sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))


import numpy
import torch
import random

from .abstract_game import AbstractGame
mydir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent = os.path.dirname(mydir)
sys.path.insert(0,parent)
import DataSet


#globals 

g_nStocks = 1
g_nFeatures = 21
g_nPeriodsInDay = 14

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        numChannels = 1
        #We have nStocks with nFeatues each plus a field to say how much we own
        self.observation_shape = (1, g_nStocks, g_nFeatures)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        #for each stock we can buy or sell we can also do nothing
        self.action_space = list(range(2*g_nStocks+1))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 10  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class


        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 126  # Maximum number of moves if game is not finished before
        self.num_simulations = 10  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 20  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 16  # Number of blocks in the ResNet
        self.channels = 256  # Number of channels in the ResNet
        self.reduced_channels_reward = 256  # Number of channels in reward head
        self.reduced_channels_value = 256  # Number of channels in value head
        self.reduced_channels_policy = 256  # Number of channels in policy head
        self.resnet_fc_reward_layers = [256,256]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [256,256]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [256,256]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = True if torch.cuda.is_available() else False  # Train on GPU if available
        if self.train_on_gpu:
            print("On GPU")
        else:
            print("No GPU found")

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0064  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer
        self.replay_buffer_size = 100  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 7  # Number of game moves to keep for every batch element
        self.td_steps = 31  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0.1  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


# In[10]:


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
      self.env = ATEnv()
      if seed<0:
          self.env.test_active(1)
          seed=None
    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return [[observation]], reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()
        #return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return [[self.env.reset()]]

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = dict()
        actions[0] = "Hold"
        for i in range(g_nStocks):
            actions[i+1] = "Buy "+str(i)
            actions[i+g_nStocks+1] = "Sell "+str(i)
        return f"{action_number}. {actions[action_number]}"

# In[15]:


class ATEnv:
    def __init__(self):
        self.data = DataSet.DataSet()
        #call AT to generate the list of closing prices for each stock
        self.closes = [self.data.getPrices(True)]
        #get the features as a list of one DF per stock make sure the order is the same
        self.features = [self.data.getFeatures(True)]
        self.max = self.data.getSize(True)-1
        self.ownership = [0]*g_nStocks
        self.time =random.randrange(1, self.max)
        print("Starting Env with max time = "+str(self.max)+ " time = "+str(self.time))
        print(self.features[0].head())
        self.cash = 1.0
        self.last_action = -1
        self.totalReward = 0
        
    def legal_actions(self):
        # Initialize to all moves and then prune.
        moves = list(range(2*g_nStocks+1))
        if sum([abs(i) for i in self.ownership])>=1:
            for i in range(len(self.ownership)):
                if self.ownership[i]>=0:
                    moves.remove(i+1)
                if self.ownership[i]<=0:
                    moves.remove(i+len(self.ownership)+1)
        return moves

    def step(self, action):
        self.last_action = action
        self.time+=1
        if action ==0:
            return self.get_observation(), self.getReward(), self.time>=self.max
        elif action-1 < len(self.ownership):
            self.ownership[action-1]+=self.getChangeValue()
        else:
            self.ownership[action-len(self.ownership)-1]-=self.getChangeValue()
        return self.get_observation(),self.getReward(), self.time>=self.max
    
    def getChangeValue(self):
        return 1.0
        #if sum([abs(i) for i in self.ownership])<1:
            #return 0.2
        #else:
            #return 0.2*sum([abs(i) for i in self.ownership])
                
    def getReward(self):
        total = 0
        #print("Reward = "+str(total)+"\nOwnership: "+str(self.ownership)+"\nLast action: "+str(self.last_action)+"\nCloses: "+str([self.closes[0][self.time],self.closes[0][self.time-1]]))
        for i in range(len(self.ownership)):
            base = self.closes[i].iloc[[self.time-1]].values[0][0]
            #print("base: "+str(base))
            if base==0:
                print("Base 0 at "+str(self.time-1))
                total=0
            else:
                total += self.ownership[i]*(self.closes[i].iloc[[self.time-1]].values[0][1]-base)/base
        self.totalReward += 100*total
        return 100*total
            
    def reset(self):
        self.ownership = [0 for i in range(g_nStocks)]
        self.time = random.randrange(1, self.max) 
        self.last_action = -1
        self.cash = 1.0
        self.totalReward = 0
        return self.get_observation()

    def test_active(self,status):
        if status:
            self.closes = [self.data.getPrices(False)]
            self.features = [self.data.getFeatures(False)]
            self.max = self.data.getSize(False)-1
            self.reset()

    def render(self):
        #TODO
        print("Total position: "+str(self.ownership)+" reward: "+str(self.getReward())+" totalReward: "+str(self.totalReward)+" actions: "+str(self.legal_actions()))

    def get_observation(self):
        #vector of features for each stock plus how much we own
        observation = numpy.zeros((g_nStocks,g_nFeatures))
        #print("Getting observation at time = "+str(self.time))
        for i in range(len(self.ownership)):
            #print("time: "+str(self.time)+"\nmax: "+str(self.max))
            features  = self.features[i].iloc[[self.time]].values.flatten().tolist()
            #print("features: "+str(features))
            #print("Ownership: "+str(self.ownership[i]))
            features.append(self.ownership[i])
            #print("Total: "+str(features))
            observation[i] = features
        return observation.flatten()
