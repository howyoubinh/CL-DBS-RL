from collections import namedtuple, deque
import random

# Define the Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """
    Replay memory for Deep Q-Learning to store experience transitions.
    
    This implementation uses a deque with a fixed capacity to efficiently
    store and sample transitions for experience replay.
    """
    def __init__(self, capacity):
        """
        Initialize the replay memory with the given capacity.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """
        Add a transition to the memory.
        
        Args:
            *args: Components of the transition (state, action, next_state, reward)
        """
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions from memory.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            list: Batch of randomly sampled transitions
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """
        Get the current size of the memory.
        
        Returns:
            int: Number of transitions in memory
        """
        return len(self.memory)