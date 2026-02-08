import os
import sys

def get_example_config(dir_path):
    config = {
        'x0_center': 'None',
        'x0_range': '1.0',
        'u_center': 'None',
        'u_range': '1.0',
        'target_state': 'None',
        'model_init': 'model = template_model()'
    }

    if "CSTR" in dir_path:
        op_point = "[0.8, 0.5, 134.14, 130.0]"
        config['x0_center'] = op_point
        config['x0_range'] = "0.1"
        config['target_state'] = op_point
        config['u_center'] = "[-10.0, 50.0]"
        config['u_range'] = "20.0"

    elif "Lotka_Volterra" in dir_path:
        op_point = "[0.5, 0.7]"
        config['x0_center'] = op_point
        config['x0_range'] = "0.1"
        config['target_state'] = op_point
        config['u_center'] = "[0.5]"
        config['u_range'] = "0.5"

    elif "batch_reactor" in dir_path:
        op_point = "[1.0, 0.5, 0.0, 120.0]"
        config['x0_center'] = op_point
        config['x0_range'] = "0.1"
        config['target_state'] = op_point
        config['u_center'] = "[0.5]"
        config['u_range'] = "0.5"

    elif "oscillating_masses" in dir_path:
        op_point = "[0.0, 0.0, 0.0, 0.0]"
        config['x0_center'] = op_point
        config['x0_range'] = "0.5"
        config['target_state'] = op_point
        config['u_center'] = "[0.0]"
        config['u_range'] = "1.0"

    elif "pendulum" in dir_path:
        op_point = "[0.0, 0.0, 0.0, 0.0]"
        if "double_inverted_pendulum" in dir_path:
             op_point = "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
             config['model_init'] = "model = template_model(obstacles=[{'x': 100, 'y': 100, 'r': 0.1}])"

        config['x0_center'] = op_point
        config['x0_range'] = "0.5"
        config['target_state'] = op_point

    elif "triple_tank" in dir_path:
        op_point = "[10.0, 10.0, 10.0]"
        config['x0_center'] = op_point
        config['x0_range'] = "2.0"
        config['target_state'] = op_point
        config['u_center'] = "[0.5, 0.5]"
        config['u_range'] = "0.5"

    return config

def generate_main_koopman(dir_path):
    print(f"Generating main_koopman.py in {dir_path}")

    abs_dir = os.path.abspath(dir_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    rel_path_to_root = os.path.relpath(repo_root, abs_dir)
    rel_path_to_examples = os.path.relpath(os.path.dirname(__file__), abs_dir)
    config = get_example_config(dir_path)

    content = f"""
import sys
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, r'{rel_path_to_root}'))
sys.path.append(os.path.join(script_dir, r'{rel_path_to_examples}'))

import do_mpc
from koopman_pdp_lib import *
from do_mpc_adapter import DoMPCAdapter

try:
    sys.path.append(script_dir)
    from template_model import template_model
    from template_simulator import template_simulator
except ImportError as e:
    print(f"Failed to import template_model/simulator: {{e}}")
    sys.exit(1)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Koopman MPC PDP Example for {{os.path.basename(script_dir)}}")

    try:
        {config['model_init']}
        simulator = template_simulator(model)
    except Exception as e:
        logger.error(f"Failed to initialize do_mpc: {{e}}")
        traceback.print_exc()
        return

    adapter = DoMPCAdapter(model, simulator)

    # Configuration
    x0_center = {config['x0_center']}
    x0_range = {config['x0_range']}
    u_center = {config['u_center']}
    u_range = {config['u_range']}
    target_state = {config['target_state']}

    # 1. Generate Data
    logger.info("--- Stage 1: Generating Data ---")
    try:
        state_trajs, control_trajs, _ = adapter.generate_trajectories(
            10, 50, x0_center=x0_center, x0_range=x0_range, u_center=u_center, u_range=u_range
        )
    except Exception as e:
        logger.error(f"Failed to generate trajectories: {{e}}")
        traceback.print_exc()
        return

    # 2. Identify Koopman
    logger.info("--- Stage 2: Identification ---")
    try:
        A_k, B_k, C_k, phi_k, n_k = identify_koopman_model(adapter, state_trajs, control_trajs)
        koopman_model = (A_k, B_k, C_k, phi_k, n_k)
    except Exception as e:
        logger.error(f"ID failed: {{e}}")
        traceback.print_exc()
        return

    # 3. Init MPC
    logger.info("--- Stage 3: Init MPC ---")
    n_theta = adapter.n_state_obs + adapter.n_control
    theta_log = np.zeros(n_theta)

    control_limits = None
    if u_center is not None:
        uc = np.array(u_center).flatten()
        ur = np.array(u_range).flatten()
        if len(uc)==1 and adapter.n_control>1: uc = np.repeat(uc, adapter.n_control)
        if len(ur)==1 and adapter.n_control>1: ur = np.repeat(ur, adapter.n_control)
        control_limits = ((uc-ur).tolist(), (uc+ur).tolist())

    try:
        mpc = KoopmanMPC(
            {{'n_state_obs': adapter.n_state_obs, 'n_control': adapter.n_control, 'dt': adapter.dt}},
            koopman_model, 10, theta_log, control_limits=control_limits, target_state=target_state
        )
    except Exception as e:
        logger.error(f"MPC init failed: {{e}}")
        traceback.print_exc()
        return

    # 4. Meta-Learning
    logger.info("--- Stage 4: Meta-Learning ---")
    x0_meta = np.array(x0_center).flatten() if x0_center is not None else np.random.uniform(-0.5,0.5,adapter.n_state_obs)
    logger.info(f"Target: {{target_state}}")

    try:
        theta_learned, meta_hist, theta_hist = learn_mpc_cost_params_pdp(
            adapter, mpc, 20, 5, 1e-4, x0_meta, target_state_meta=target_state
        )
        logger.info(f"Learned theta: {{theta_learned}}")
    except Exception as e:
        logger.error(f"Meta-learning failed: {{e}}")
        traceback.print_exc()
        return

    # 5. Visualize final performance
    logger.info("--- Stage 5: Visualization ---")
    # Run a simulation with learned theta
    x = x0_meta.copy()
    hist_x = [x]
    hist_u = []
    for _ in range(50):
        res = mpc.solve_mpc_step(x, theta_learned)
        if res['success']:
            u = res['control_traj_opt'][0]
        else:
            u = np.zeros(adapter.n_control)
        x = adapter.dynamics_step(x, u)
        hist_x.append(x)
        hist_u.append(u)

    plot_results(adapter, hist_x, hist_u, title=f"{{os.path.basename(script_dir)}}_Final")
    logger.info("Done.")

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
