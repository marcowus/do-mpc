import os
import sys

def get_example_config(dir_path):
    config = {
        'x0_center': 'None',
        'x0_range': '1.0',
        'u_center': 'None',
        'u_range': '1.0',
        'model_init': 'model = template_model()'
    }

    # Simple heuristic based on directory name
    if "CSTR" in dir_path:
        # C_a, C_b, T_R, T_K
        config['x0_center'] = "[0.8, 0.5, 134.14, 130.0]"
        config['x0_range'] = "0.1"
        # u: Q_dot, F. F must be positive.
        config['u_center'] = "[-10.0, 50.0]"
        config['u_range'] = "20.0" # Q: [-30, 10], F: [30, 70]
    elif "Lotka_Volterra" in dir_path:
        # x_0, x_1
        config['x0_center'] = "[0.5, 0.7]"
        config['x0_range'] = "0.1"
        # u: inp > 0?
        config['u_center'] = "[0.5]"
        config['u_range'] = "0.5"
    elif "batch_reactor" in dir_path:
        # X_s, S_s, P_s, V_s
        config['x0_center'] = "[1.0, 0.5, 0.0, 120.0]"
        config['x0_range'] = "0.1"
        # u: inp > 0
        config['u_center'] = "[0.5]"
        config['u_range'] = "0.5"
    elif "oscillating_masses" in dir_path:
        # x1, v1, x2, v2
        config['x0_center'] = "[0.0, 0.0, 0.0, 0.0]"
        config['x0_range'] = "0.5"
        config['u_center'] = "[0.0]"
        config['u_range'] = "1.0"
    elif "double_inverted_pendulum" in dir_path:
        # 6 states: pos, theta1, theta2, dpos, dtheta1, dtheta2
        config['x0_center'] = "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
        config['x0_range'] = "0.5"
        config['model_init'] = "model = template_model(obstacles=[{'x': 100, 'y': 100, 'r': 0.1}])"
    elif "pendulum" in dir_path:
        # pos, vel, pos, vel (Simple pendulum is usually 2 or 4 states depending on implementation)
        # double_inverted_pendulum is handled above.
        # Check standard pendulum if exists?
        # Assuming 4 for general case or check n_x later.
        # Actually safer to not set x0_center if unknown n_x, but we set 'None' by default.
        # But 'pendulum' block sets it to 4 zeros.
        # If there is another pendulum example with different states, it might fail.
        # Let's revert default 'pendulum' block to use None if not sure.
        # But for 'double_inverted_pendulum' we need 6.
        pass

    elif "triple_tank" in dir_path:
        # h1, h2, h3
        config['x0_center'] = "[10.0, 10.0, 10.0]"
        config['x0_range'] = "2.0"
        # u: pumps > 0
        config['u_center'] = "[0.5, 0.5]"
        config['u_range'] = "0.5"

    return config

def generate_main_koopman(dir_path):
    print(f"Generating main_koopman.py in {dir_path}")

    # Calculate relative path to root (where do_mpc is)
    abs_dir = os.path.abspath(dir_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # examples/.. -> root

    rel_path_to_root = os.path.relpath(repo_root, abs_dir)
    rel_path_to_examples = os.path.relpath(os.path.dirname(__file__), abs_dir)

    config = get_example_config(dir_path)

    content = f"""
import sys
import os
import numpy as np
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))

# Add do_mpc to path
sys.path.append(os.path.join(script_dir, r'{rel_path_to_root}'))
# Add examples folder to path (for lib and adapter)
sys.path.append(os.path.join(script_dir, r'{rel_path_to_examples}'))

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
    print(f"Failed to import template_model/simulator: {{e}}")
    sys.exit(1)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting Koopman MPC PDP Example for {{os.path.basename(script_dir)}}")

    # 1. Setup do_mpc model and simulator
    try:
        {config['model_init']}
        simulator = template_simulator(model)
    except Exception as e:
        logger.error(f"Failed to initialize do_mpc model/simulator: {{e}}")
        return

    # 2. Wrap in adapter
    adapter = DoMPCAdapter(model, simulator)

    # 3. Generate Data for Koopman ID
    NUM_TRAJ_KOOPMAN = 10
    TRAJ_LEN_KOOPMAN = 50
    logger.info("--- Stage 1: Generating Data for Koopman ID ---")

    # Example-specific configuration
    x0_center = {config['x0_center']}
    x0_range = {config['x0_range']}
    u_center = {config['u_center']}
    u_range = {config['u_range']}

    try:
        state_trajs_obs, control_trajs, _ = adapter.generate_trajectories(
            NUM_TRAJ_KOOPMAN, TRAJ_LEN_KOOPMAN,
            x0_center=x0_center, x0_range=x0_range,
            u_center=u_center, u_range=u_range
        )
    except Exception as e:
        logger.error(f"Failed to generate trajectories: {{e}}")
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
        logger.error(f"Koopman identification failed: {{e}}")
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

    system_params_dict = {{
        'n_state_obs': adapter.n_state_obs,
        'n_control': adapter.n_control,
        'dt': adapter.dt
    }}

    # Determine control limits for MPC from u_center/range if available, or defaults
    if u_center is not None and u_range is not None:
        uc = np.array(u_center).flatten()
        ur = np.array(u_range).flatten()
        # If scalar provided but multiple controls, broadcast
        if len(uc) == 1 and adapter.n_control > 1:
            uc = np.repeat(uc, adapter.n_control)
        if len(ur) == 1 and adapter.n_control > 1:
            ur = np.repeat(ur, adapter.n_control)

        control_lb = (uc - ur).tolist()
        control_ub = (uc + ur).tolist()
        control_limits = (control_lb, control_ub)
    else:
        control_limits = None

    logger.info("--- Stage 3: Initialising Koopman MPC Controller ---")
    try:
        koopman_mpc = KoopmanMPC(system_params_dict, koopman_model_tuple, MPC_HORIZON, initial_theta_cost_log_weights, control_limits=control_limits)
    except Exception as e:
        logger.error(f"Failed to initialize Koopman MPC: {{e}}")
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
        logger.info(f"Learned theta: {{learned_theta}}")
    except Exception as e:
        logger.error(f"Meta-learning failed: {{e}}")
        return

    logger.info("Example completed successfully.")

if __name__ == "__main__":
    main()
"""

    output_path = os.path.join(dir_path, "main_koopman.py")
    with open(output_path, "w") as f:
        f.write(content)

def find_and_deploy():
    root_examples = os.path.dirname(os.path.abspath(__file__))

    for root, dirs, files in os.walk(root_examples):
        if "template_model.py" in files and "template_simulator.py" in files:
            generate_main_koopman(root)

if __name__ == "__main__":
    find_and_deploy()
