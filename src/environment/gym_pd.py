import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from src.simulation.simulate_network_optimized import simulate_network_opt as simulate_network
from src.utils.data_processing import constructor_hash

class MousePDEnv(gym.Env):
    """
    Gymnasium environment for Parkinson's Disease Deep Brain Stimulation simulation.

    This environment allows reinforcement learning agents to control DBS parameters
    (frequency, pulse width, amplitude) to reduce pathological oscillations in the
    basal ganglia while minimizing energy consumption.

    Supports both discrete and continuous action spaces.
    """
    def __init__(self, leap: dict, num_steps: int, tau_beta_max: float, tau_reward: float,
                 delta: float, max_steps: int, N: int=10, seed: int=None, pd: int=1,
                 corstim: int=0, TMAX: int=100, dt: float=0.01, it: int=0, thresh_time_req: int=5,
                 action_type: str = 'discrete'): # Added action_type parameter
        """
        Initialize the MousePD environment.

        Args:
            leap (dict): Scaling factors for DBS parameters
            num_steps (int): Size of output array of spike times
            tau_beta_max (float): Target threshold for beta oscillations
            tau_reward (float): Reward scaling factor
            delta (float): Energy penalty factor
            max_steps (int): Maximum number of steps per episode
            N (int): Number of neurons in each population
            seed (int): Random seed
            pd (int): Flag for Parkinson's disease condition (0: healthy, 1: PD)
            corstim (int): Flag for cortical stimulation
            TMAX (int): Maximum simulation time
            dt (float): Time step size in ms
            it (int): Iteration number
            thresh_time_req (int): Number of consecutive steps under threshold to terminate
            action_type (str): Type of action space ('discrete' or 'continuous'). Default: 'discrete'.
        """
        super(MousePDEnv, self).__init__()

        # Validate action_type
        if action_type not in ['discrete', 'continuous']:
            raise ValueError(f"Invalid action_type '{action_type}'. Must be 'discrete' or 'continuous'.")
        self.action_type = action_type

        # Define class variables
        self.leap = leap # Used only for discrete actions
        self.seed = seed
        self.num_steps = num_steps
        self.freq = 40           # Default from matlab; range:[0,200]
        self.pw = 0.3            # Default pulse width
        self.amp = 300           # Default amplitude
        self.t = 0               # Current step

        self.tau_beta_max = tau_beta_max
        self.tau_reward = tau_reward 
        self.delta = delta 
        self.max_steps = max_steps
        self.N = N 

        self.thresh_time = 0 
        self.thresh_time_req = thresh_time_req 
        self.step_count = 0 
        self.Idbs = 0.
        
        # Simulation constants
        self.PD = pd
        self.CORSTIM = corstim
        self.TMAX = TMAX
        self.DT = dt
        self.it = it

        # Define action and observation spaces based on action_type
        if self.action_type == 'discrete':
            self.action_space = spaces.Dict({
                'freq': spaces.Discrete(3, start=-1),  # [-1, 0, 1]
                'pw': spaces.Discrete(3, start=-1),    # [-1, 0, 1]
                'amp': spaces.Discrete(3, start=-1)    # [-1, 0, 1]
            })
        elif self.action_type == 'continuous':
            # Define continuous action bounds (slightly narrowed to improve stability)
            self.action_low = np.array([0.0, 0.05, 0.0], dtype=np.float32)  # Freq (Hz), PW (ms), Amp (nA/cm^2) - Increased PW lower bound
            self.action_high = np.array([200.0, 1.0, 550.0], dtype=np.float32) # Example bounds, adjust as needed - Decreased Amp upper bound
            self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)

        # Observation space
        # Calculate total neurons based on constructor_hash output
        # Assuming N per region and 8 regions (TH, STN, GPe, GPi, Str_indr, Str_dr, Cor_E, Cor_I)
        num_regions = 8 # Adjust if constructor_hash changes
        total_neurons = self.N * num_regions # Should be 10 * 8 = 80
        self.observation_space = spaces.Dict({
            'spikes': spaces.Box(low=0, high=1, shape=(self.num_steps, total_neurons)), # Use total_neurons
            'E': spaces.Box(low=0, high=np.inf, shape=(1,))
        })
        
    def set_pd_state(self, is_pd):
        """
        Set the Parkinson's Disease state.
        is_pd: 1 for PD (pathological), 0 for Healthy.
        """
        self.PD = int(is_pd)

    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            tuple: Initial observation (spikes, energy)
        """
        if self.seed is None:
            self.seed = random.randint(0, 10000)
        
        self.step_count = 0 
        
        # Run the simulation with default parameters
        sim_vars = simulate_network(
            self.it, self.PD, self.CORSTIM, self.N, self.TMAX, self.DT, seed=self.seed
        )
        self.set_data(sim_vars)
        self.terminated = False
        self.truncated = False

        # default DBS starting params:
        self.freq = 40
        self.pw = 0.3
        self.amp = 300
        self.thresh_time = 0

        return self.spikes, self.E

    def step(self, action):
        """
        Take a step in the environment with the given action, handling both discrete and continuous types.

        Args:
            action: The action from the agent. Type depends on self.action_type:
                    - 'discrete': dict with keys 'freq', 'pw', 'amp' and values [-1, 0, 1]
                    - 'continuous': np.ndarray with shape (3,) for [frequency, pulse_width, amplitude]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if self.action_type == 'discrete':
            # --- Discrete Action Handling ---
            if not isinstance(action, dict):
                raise TypeError(f"Expected action of type dict for discrete mode, got {type(action)}")
            act_freq = action.get('freq', 0)
            act_pw = action.get('pw', 0)
            act_amp = action.get('amp', 0)
            # Update parameters based on discrete steps and leap values
            new_freq = self.freq + (act_freq * self.leap['freq'])
            new_pw = self.pw + (act_pw * self.leap['pw'])
            new_amp = self.amp + (act_amp * self.leap['amp'])

            # Clip to realistic bounds
            # Freq: 0 to 180 Hz (Saturates at 150, buffer to 180)
            self.freq = np.clip(new_freq, 0.0, 180.0)
            
            # PW: 0.06 to 0.4 ms (Do not go below 0.06 or physics breaks)
            self.pw = np.clip(new_pw, 0.06, 0.4)
            
            # Amp: 0 to 250 uA/cm^2 (Above 250 is wasted energy)
            self.amp = np.clip(new_amp, 0.0, 250.0)

        elif self.action_type == 'continuous':
            # --- Continuous Action Handling ---
            if not isinstance(action, np.ndarray):
                 raise TypeError(f"Expected action of type np.ndarray for continuous mode, got {type(action)}")
            # Clip the continuous actions to ensure they are within valid bounds
            clipped_action = np.clip(action, self.action_low, self.action_high)
            # Directly assign clipped continuous values
            self.freq = clipped_action[0]
            self.pw = clipped_action[1]
            self.amp = clipped_action[2]

        self.step_count += 1

        # Store the applied stimulation parameters
        self.stim_set = {
            'freq': self.freq,
            'pw': self.pw,
            'amp': self.amp,
        }

        # Run simulation with updated parameters
        sim_vars = simulate_network(
            self.it, self.PD, self.CORSTIM, self.N, self.TMAX, self.DT, 
            PW=self.pw, amplitude=self.amp, dbs_freq=self.freq, 
            states=self.states, seed=self.seed
        )
        
        self.set_data(sim_vars)
        
        # Check if below threshold for required time
        if self.under_threshold():
            self.thresh_time += 1 
        else: 
            self.thresh_time = 0 
        
        if self.thresh_time >= self.thresh_time_req: 
            self.terminated = True

        # Check for max steps
        if self.t > self.max_steps:
            self.truncated = True

        # Calculate reward
        self.calc_reward()

        return self.spikes, self.reward, self.terminated, self.truncated, self.stim_set 
    
    def under_threshold(self):
        """
        Check if beta oscillation power is below threshold.
        
        Returns:
            bool: True if below threshold, False otherwise
        """
        return self.gpi_alpha_beta_area < self.tau_beta_max
        
    def set_data(self, sim_vars): 
        """
        Set internal state from simulation variables.
        
        Args:
            sim_vars (dict): Simulation results
        """
        self.gpi_alpha_beta_area = sim_vars['gpi_alpha_beta_area']
        self.states = sim_vars['states']
        self.Idbs = sim_vars['Idbs']
        # Process spike times into format for NN
        self.spikes = constructor_hash(sim_vars, num_steps=self.num_steps)
        self.E = self.energy()

    def calc_reward(self): 
        """
        Calculate reward based on beta oscillation power and energy consumption.
        """
        # Energy penalty
        w_energy = self.delta * self.E
        
        if not self.terminated:
            # Calculate reward based on distance from threshold
            dist = (self.gpi_alpha_beta_area - self.tau_beta_max)**2
            
            if self.under_threshold():
                # Reward for being under threshold
                self.reward = self.tau_reward - w_energy
            else: 
                # Penalty proportional to distance from threshold
                self.reward = -dist - w_energy 
        else:
            # Bonus reward for achieving termination condition
            remaining_steps = self.max_steps - self.step_count + 1
            self.reward = (self.tau_reward * remaining_steps) - w_energy
            
        self.reward = float(self.reward)

    def energy(self):
        """
        Calculate energy consumption based on DBS current.
        
        Returns:
            float: Energy consumption
        """
        return np.sqrt(np.mean(np.square(self.Idbs)))
    
    def render(self):
        """
        Render the current state of the environment.
        """
        print(self.spikes)

# Class for Xylo

class MousePDEnvXylo(MousePDEnv):
    
    def reset(self, seed=None, options=None):
        """
        Overrides the original reset to return a (obs, info) tuple.
        The actual reset logic is handled by the parent class.
        """
        # Call the parent's reset method to get the original outputs
        initial_observation, initial_E = super().reset()
        
        # In this context, the initial_spikes is our raw data dictionary
        raw_spike_output = initial_observation
        
        # Create a dummy observation (can be empty, it won't be used)
        dummy_observation = np.zeros_like(self.observation_space.sample())

        # Package the real data into the info dictionary
        info = {'raw_spike_data': raw_spike_output}

        return dummy_observation, info

    def step(self, action):
        """
        Overrides the original step to return a 5-element tuple with an info dict.
        The actual step logic is handled by the parent class.
        """
        # Call the parent's step method to get the original outputs
        observation, reward, terminated, truncated, original_info = super().step(action)
        
        # Extract only the spike matrix (the first element)
        raw_spike_output = observation
        
        # Create a dummy observation
        dummy_observation = np.zeros_like(self.observation_space.sample())
        
        # Package the real data into the new info dictionary
        info = {'raw_spike_data': raw_spike_output}

        return dummy_observation, reward, terminated, truncated, info
    
class MousePDEnvSoftReward(MousePDEnvXylo):
    def calc_reward(self):
        """
        Soft Reward Implementation with Hinge Loss:
        - Beta > Threshold: Penalty proportional to error.
        - Beta <= Threshold: Flat reward plateau, allowing Energy to dominate optimization.
        """
        # 1. Energy Penalty
        w_energy = self.delta * self.E

        # 2. Hinge Error (Only penalize if ABOVE threshold)
        # If Beta < Threshold, error is 0.
        # If Beta > Threshold, error is positive.
        error = max(0, self.gpi_alpha_beta_area - self.tau_beta_max)
        
        reward_scale = 30.0 
        
        # 3. Reward Structure
        # Base Reward (tau_reward, e.g. 3000) - Error Penalty - Energy
        step_reward = self.tau_reward - (error * reward_scale) - w_energy

        if not self.terminated:
            self.reward = step_reward
        else:
            # Project forward
            remaining_steps = self.max_steps - self.step_count + 1
            self.reward = step_reward * remaining_steps
            
        self.reward = float(self.reward)

class MousePDEnvAdaptive(MousePDEnvXylo):
    """
    Adaptive Reward for energy-aware DBS control.
    
    Restructures the reward so that energy savings are meaningful:
    - β > threshold: Penalize proportionally to error + moderate energy penalty
    - β ≤ threshold: Reward scales with energy savings (0 energy → max reward)
    
    This creates three emergent behaviors:
    1. PD + β high → increase stimulation (suppress oscillations)
    2. PD + β low → reduce to minimum effective dose (energy savings dominate)
    3. Healthy + β low → turn off stimulation (max reward at zero energy)
    """
    def calc_reward(self):
        # Normalize energy to [0, 1] range
        # E_max ≈ 250 is approximate RMS of Idbs at max stimulation
        E_max = 250.0
        E_normalized = min(self.E / E_max, 1.0)
        
        # Hinge error: only penalize if β is ABOVE threshold
        error = max(0, self.gpi_alpha_beta_area - self.tau_beta_max)
        
        reward_scale = 30.0
        
        # alpha controls how much energy matters in the below-threshold branch
        # Capped at 1.0 to prevent reward from going negative
        alpha = min(self.delta, 1.0)
        
        if error > 0:
            # β above threshold: pure suppression penalty, no energy component, so
            # the only way to gain positive reward is to cross below threshold.
            step_reward = -(error * reward_scale)
        else:
            # β below threshold: reward = τ × ((1-α) + α × energy_savings).
            # δ (alpha) is the single knob controlling the suppression/energy tradeoff:
            # δ=0.01 → reward ≈ τ; δ=0.5 → τ × (0.5 + 0.5 × savings); δ=1.0 → τ × savings.
            energy_savings = 1.0 - E_normalized
            step_reward = self.tau_reward * ((1.0 - alpha) + alpha * energy_savings)

        # Per-step reward with no termination bonus, so the agent maximizes cumulative
        # reward over the full episode rather than ending early.
        self.reward = float(step_reward)