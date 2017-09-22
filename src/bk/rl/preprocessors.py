"""Suggested Preprocessors."""

import numpy as np

from rl import utils
from rl.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for seeing the trend of the change, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self._history_length = history_length;
        self._flag_start_net = True;
        self._stacked_return_net = None;
        self._flag_start_mem = True;
        self._stacked_return_mem = None;

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take.
        
        state: np.ndarray, one dimension
          Expect a one dimension array for state features 
        """
        if self._flag_start_net:
            self._stacked_return_net = np.zeros((state.size
                                             ,self._history_length));
            self._stacked_return_net[:,-1] = state;
            self._flag_start_net = False;
        else:
            for i in range(self._history_length - 1):
                self._stacked_return_net[:,i] = \
                    self._stacked_return_net[:,i + 1];
            self._stacked_return_net[:,-1] = state;
            
        return np.copy(self._stacked_return_net.reshape((1,) 
                                               + self._stacked_return_net.shape));

    def process_state_for_memory(self, state):
        """You only want history when you're deciding the current action to take.
        
        state: np.ndarray, one dimension
          Expect a one dimension array for observation features 
        """
        if self._flag_start_mem:
            self._stacked_return_mem = np.zeros((state.size
                                             ,self._history_length));
            self._stacked_return_mem[:,-1] = state;
            self._flag_start_mem = False;
        else:
            for i in range(self._history_length - 1):
                self._stacked_return_mem[:,i] = \
                    self._stacked_return_mem[:,i + 1];
            self._stacked_return_mem[:,-1] = state;
            
        return np.copy(self._stacked_return_mem.reshape((1,) 
                                               + self._stacked_return_mem.shape));

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self._flag_start_net = True;
        self._flag_start_mem = True;

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        pass

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        pass

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        pass

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        pass

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        pass


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        pass
