import numpy as np
import logging

logger = logging.getLogger(__name__)

class DoMPCAdapter:
    """
    Adapter to wrap do_mpc model and simulator for use with Koopman PDP library.
    """
    def __init__(self, model, simulator):
        self.model = model
        self.simulator = simulator

        # Dimensions
        self.n_state_obs = self.model.n_x
        self.n_control = self.model.n_u

        # Time step
        self.dt = self.simulator.settings.t_step

        # Initial state (placeholder)
        self.x0 = np.zeros((self.n_state_obs, 1))

        # Control bounds (optional, if model has them)
        # do_mpc models store bounds in _x_lb, _x_ub, _u_lb, _u_ub dictionaries usually
        # but accessed via specific methods or properties if available.
        # For now we don't strictly enforce them here as the KoopmanMPC sets its own.

    def dynamics_step(self, x_current, u_current, p_current=None):
        """
        Step the system dynamics.
        x_current: (n_state,)
        u_current: (n_control,)
        Returns: x_next (n_state,)
        """
        # Reshape for do_mpc (expects column vectors usually)
        if x_current.ndim == 1:
            x_curr_reshaped = x_current.reshape(-1, 1)
        else:
            x_curr_reshaped = x_current

        if u_current.ndim == 1:
            u_curr_reshaped = u_current.reshape(-1, 1)
        else:
            u_curr_reshaped = u_current

        # Set state in simulator
        self.simulator.x0 = x_curr_reshaped

        # Make step
        # simulator.make_step(u0) returns x_next
        x_next = self.simulator.make_step(u_curr_reshaped)

        return x_next.flatten()

    def generate_trajectories(self, num_traj, traj_len, x0_center=None, x0_range=1.0, u_center=None, u_range=1.0):
        """
        Generate random trajectories for Koopman ID.
        """
        state_traj_list = []
        control_traj_list = []

        logger.info(f"Generating {num_traj} trajectories of length {traj_len}...")

        if x0_center is None:
            x0_center = np.zeros((self.n_state_obs, 1))

        if isinstance(x0_center, (list, np.ndarray)):
            x0_center = np.array(x0_center).reshape(self.n_state_obs, 1)

        if u_center is None:
            u_center = np.zeros((self.n_control, 1))

        if isinstance(u_center, (list, np.ndarray)):
            u_center = np.array(u_center).reshape(self.n_control, 1)

        for i in range(num_traj):
            # Random initial state around center
            x0 = x0_center + np.random.uniform(-x0_range, x0_range, (self.n_state_obs, 1))

            # Reset simulator
            self.simulator.x0 = x0
            self.simulator.reset_history()

            traj_x = [x0.flatten()]
            traj_u = []

            for _ in range(traj_len):
                # Random control around center
                u_k = u_center + np.random.uniform(-u_range, u_range, (self.n_control, 1))

                x_next = self.simulator.make_step(u_k)

                traj_x.append(x_next.flatten())
                traj_u.append(u_k.flatten())

            state_traj_list.append(np.array(traj_x))
            control_traj_list.append(np.array(traj_u))

        return state_traj_list, control_traj_list, None
