
import sys
import os
import numpy as np
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))

# Add do_mpc to path
sys.path.append(os.path.join(script_dir, r'../../..'))
# Add examples folder to path (for lib and adapter)
sys.path.append(os.path.join(script_dir, r'../..'))

import do_mpc
from koopman_pdp_lib import *
from do_mpc_adapter import DoMPCAdapter

# Local imports
try:
    # Use context manager to temporarily add script_dir to sys.path for local imports if needed
    sys.path.append(script_dir)
    from template_model import template_model
    from template_simulator import template_simulator
except ImportError as e:
    print(f"Failed to import template_model/simulator: {e}")
    sys.exit(1)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting Koopman MPC PDP Example for {os.path.basename(script_dir)}")

    # 1. Setup do_mpc model and simulator
    try:
        model = template_model()
        simulator = template_simulator(model)
    except Exception as e:
        logger.error(f"Failed to initialize do_mpc model/simulator: {e}")
        return

    # 2. Wrap in adapter
    adapter = DoMPCAdapter(model, simulator)

    # 3. Generate Data for Koopman ID
    NUM_TRAJ_KOOPMAN = 10
    TRAJ_LEN_KOOPMAN = 50
    logger.info("--- Stage 1: Generating Data for Koopman ID ---")

    # Example-specific configuration
    x0_center = [0.8, 0.5, 134.14, 130.0]
    x0_range = 0.1

    try:
        state_trajs_obs, control_trajs, _ = adapter.generate_trajectories(
            NUM_TRAJ_KOOPMAN, TRAJ_LEN_KOOPMAN, x0_center=x0_center, x0_range=x0_range
        )
    except Exception as e:
        logger.error(f"Failed to generate trajectories: {e}")
        return

    # 4. Identify Koopman Model
    logger.info("--- Stage 2: Identifying Koopman Model ---")
    try:
        # Using default phi_func (quadratic)
        A_k, B_k, C_k, phi_k, n_k_lifted = identify_koopman_model(
            adapter, state_trajs_obs, control_trajs
        )
        koopman_model_tuple = (A_k, B_k, C_k, phi_k, n_k_lifted)
    except Exception as e:
        logger.error(f"Koopman identification failed: {e}")
        return

    # 5. Initialize MPC
    MPC_HORIZON = 10
    SIM_HORIZON_META = 20
    NUM_META_ITERATIONS = 5
    LEARNING_RATE_META = 1e-4

    # Initial theta cost (log weights)
    # Dimension: n_state + n_control
    n_theta = adapter.n_state_obs + adapter.n_control
    initial_theta_cost_log_weights = np.zeros(n_theta) # Start with uniform weights (log(1)=0)

    system_params_dict = {
        'n_state_obs': adapter.n_state_obs,
        'n_control': adapter.n_control,
        'dt': adapter.dt
    }

    logger.info("--- Stage 3: Initialising Koopman MPC Controller ---")
    try:
        koopman_mpc = KoopmanMPC(system_params_dict, koopman_model_tuple, MPC_HORIZON, initial_theta_cost_log_weights)
    except Exception as e:
        logger.error(f"Failed to initialize Koopman MPC: {e}")
        return

    # 6. Meta-Learning
    # Initial state for meta-learning simulation
    # Use center if available, else random
    if x0_center is not None:
        initial_true_state = np.array(x0_center).flatten()
    else:
        initial_true_state = np.random.uniform(-0.5, 0.5, adapter.n_state_obs)

    logger.info("--- Stage 4: Starting Meta-Learning ---")
    try:
        learned_theta, meta_hist, theta_hist = learn_mpc_cost_params_pdp(
            true_system=adapter,
            koopman_mpc_controller=koopman_mpc,
            sim_horizon_meta=SIM_HORIZON_META,
            num_meta_iterations=NUM_META_ITERATIONS,
            learning_rate_meta=LEARNING_RATE_META,
            initial_true_state_full_np=initial_true_state
        )
        logger.info(f"Learned theta: {learned_theta}")
    except Exception as e:
        logger.error(f"Meta-learning failed: {e}")
        return

    logger.info("Example completed successfully.")

if __name__ == "__main__":
    main()
