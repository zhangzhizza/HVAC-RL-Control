"""RL Policy classes.

We have provided you with a base policy class, some example
implementations and some unimplemented classes that should be useful
in your code.
"""
import numpy as np
import copy

ACTION_DICT = {0: (-0.5, -0.5), 1:(-0.5, 0),
              2:(-0.5, 0.5), 3:(0, -0.5),
              4:(0, 0), 5:(0, 0.5),
              6:(0.5, -0.5), 7:(0.5, 0), 
              8:(0.5, -0.5)}


class Policy:
    """Base class representing an MDP policy.

    Policies are used by the agent to choose actions.

    Policies are designed to be stacked to get interesting behaviors
    of choices. For instances in a discrete action space the lowest
    level policy may take in Q-Values and select the action index
    corresponding to the largest value. If this policy is wrapped in
    an epsilon greedy policy then with some probability epsilon, a
    random action will be chosen.
    """

    def select_action(self, **kwargs):
        """Used by agents to select actions.

        Returns
        -------
        Any:
          An object representing the chosen action. Type depends on
          the hierarchy of policy instances.
        """
        raise NotImplementedError('This method should be overriden.')

    def process_action(self, setpoint_this, action):
        """ 
          1. Process action index to action command tuple 
          2 . Process action command tuple to heating and cooling set point to HVAC
            
          Note: The heating set point should be always lower than cooling set point
        

        Parameters
        ----------
        setpoint_this: list of float 
            list[0]: current heating setpoint
            list[1]: current cooling setpoint

        action: int 
            # see ACTION_DICT at the top

        Returns
            -------
            list: (float, float) fist is heating setpoint, second for cooling setpoint

        """
        # get new set point based on action index
        setpoint_next = copy.deepcopy(setpoint_this)
   
        setpoint_next[0]  = setpoint_this[0] + ACTION_DICT.get(action)[0]
        setpoint_next[1]  = setpoint_this[1] + ACTION_DICT.get(action)[1]
        

        ##Three cases coulde make heating setpoint higher than cooling setpoint
        #case 1: heating no change and cooling decrease
        #case 2: heating increase and cooling no change
        #case 3: heating incease and cooling decrease
        if(setpoint_next[0] > setpoint_next[1]):
                # don't take any action
                setpoint_next[0] = setpoint_this[0] 
                setpoint_next[1] = setpoint_this[1] 

        return setpoint_next




class UniformRandomPolicy(Policy):
    """Chooses a discrete action with uniform random probability.

    This is provided as a reference on how to use the policy class.

    Parameters
    ----------
    num_actions: int
      Number of actions to choose from. Must be > 0.

    Raises
    ------
    ValueError:
      If num_actions <= 0
    """

    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions


    def select_action(self):
        """Return a random action index.

        The action will be processed as .

        Parameters
        ----------
        setpoint_this: list of float 
        list[0]: current heating setpoint
        list[1]: current cooling setpoint

        Returns
        -------
        action index: int 
          The index of perumuation set of ist is heating setpoint, second for cooling setpoint
        """
        
        return np.random.randint(0, self.num_actions)


    def get_config(self):  # noqa: D102
        return {'num_actions': self.num_actions}


class GreedyPolicy(Policy):
    """Always returns best action according to Q-values.

    This is a pure exploitation policy.
    """

    def select_action(self, q_values, **kwargs):  # noqa: D102
        return np.argmax(q_values)


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.

    Standard greedy-epsilon implementation. With probability epsilon
    choose a random action. Otherwise choose the greedy action.

    Parameters
    ----------
    epsilon: float
     Initial probability of choosing a random action. Can be changed
     over time.
    """
    def __init__(self, epsilon):
        self._epsilon = epsilon;

    def select_action(self, q_values, **kwargs):
        """Run Greedy-Epsilon for the given Q-values.

        Parameters
        ----------
        q_values: array-like
          Array-like structure of floats representing the Q-values for
          each action.

        Returns
        -------
        int:
          The action index chosen.
        """
        sample = np.random.uniform();
        
        greedy = np.argmax(q_values);
        uniformrand = np.random(0, q_values.shape[1]);
        
        if sample < self._epsilon:
            return uniformrand;
        else:
            return greedy;
        


class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.

    """

    def __init__(self, start_value, end_value,
                 num_steps):  # noqa: D102
        self._start_value = start_value;
        self._end_value = end_value;
        self._decay_step = 1.0 * (start_value - end_value)/num_steps;
        self._decayed_epsilon = start_value;

        
    def select_action(self, q_values, is_training, **kwargs):
        """Decay parameter and select action.

        Parameters
        ----------
        q_values: np.array
          The Q-values for each action.
        is_training: bool, optional
          If true then parameter will be decayed. Defaults to true.

        Returns
        -------
        Any:
          Selected action.
        """
        sample = np.random.uniform();
        
        greedy = np.argmax(q_values);
        uniformrand = np.random.randint(0, q_values.shape[1]);
        
        if is_training:
            self._decayed_epsilon -= self._decay_step;
            if sample < self._decayed_epsilon:
                return uniformrand;
            else:
                return greedy;
        else:
            if sample < self._end_value:
                return uniformrand;
            else:
                return greedy;
        

    def reset(self):
        """Start the decay over at the start value."""
        self._decayed_epsilon = self._start_value;
