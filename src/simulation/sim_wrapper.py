import numpy as np
from src.simulation.simulate_network_optimized import simulate_network_opt
from src.utils.data_processing import constructor_hash

class SimulationWrapper:
    """
    Wrapper for the biophysical network simulation to support discrete time-stepping
    for Hardware-in-the-Loop (HIL) verification.
    """
    def __init__(self, n_neurons=10, t_step=100, dt=0.01, pd=1, seed=None):
        """
        Initialize the simulation wrapper.

        Args:
            n_neurons (int): Number of neurons per population.
            t_step (int): Duration of each simulation step in ms.
            dt (float): Integration time step in ms.
            pd (int): Parkinson's disease condition (0: Healthy, 1: PD).
            seed (int): Random seed.
        """
        self.n = n_neurons
        self.t_step = t_step
        self.dt = dt
        self.pd = pd
        self.seed = seed
        self.states = {}
        self.pd_state = 1 # Default to PD
        self.it = 0
        self.time_elapsed = 0

    def set_pd_state(self, is_pd):
        """
        Set the Parkinson's Disease state.
        is_pd: 1 for PD (pathological), 0 for Healthy.
        """
        self.pd_state = int(is_pd)
        
    def reset(self):
        """
        Reset the simulation to initial conditions.
        """
        self.states = {}
        self.it = 0
        self.time_elapsed = 0
        # Run one initial step with no stimulation to prime the state
        return self.step(action={'freq': 0, 'pw': 0, 'amp': 0})

    def step(self, action):
        """
        Advance the simulation by one time-step.

        Args:
            action (dict): DBS parameters {'freq': float, 'pw': float, 'amp': float}.

        Returns:
            dict: {
                'spike_matrix': np.ndarray, # (num_steps, num_channels)
                'lfp': float, # Beta band power
                'raw_vars': dict # Full simulation output
            }
        """
        freq = action.get('freq', 0)
        pw = action.get('pw', 0)
        amp = action.get('amp', 0)

        # Run simulation
        sim_vars = simulate_network_opt(
            IT=self.it,
            pd=self.pd_state,
            corstim=0, # no cortical stimulation
            n=self.n,
            tmax=self.t_step,
            dt=self.dt,
            PW=pw,
            amplitude=amp,
            dbs_freq=freq,
            states=self.states,
            seed=self.seed
        )

        # Update internal state
        self.states = sim_vars['states']
        self.it += 1
        self.time_elapsed += self.t_step
        
        num_steps_out = int(self.t_step) # Assuming 1ms binning for the output matrix usually
        spike_matrix = constructor_hash(sim_vars, num_steps=num_steps_out)
        
        return {
            'spike_matrix': spike_matrix,
            'lfp': sim_vars['gpi_alpha_beta_area'],
            'raw_vars': sim_vars
        }
