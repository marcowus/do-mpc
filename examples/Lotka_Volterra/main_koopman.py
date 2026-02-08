
import sys
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, r'../..'))
sys.path.append(os.path.join(script_dir, r'..'))

import do_mpc
from koopman_pdp_lib import *
from do_mpc_adapter import DoMPCAdapter

try:
    sys.path.append(script_dir)
    from template_model import template_model
    from template_simulator import template_simulator
except ImportError as e:
    print(f"Failed to import template_model/simulator: {e}")
    sys.exit(1)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Koopman MPC PDP Example for {os.path.basename(script_dir)}")

    try:
        model = template_model()
        simulator = template_simulator(model)
    except Exception as e:
        logger.error(f"Failed to initialize do_mpc: {e}")
        traceback.print_exc()
        return

    adapter = DoMPCAdapter(model, simulator)

    # Configuration
    x0_center = [0.5, 0.7]
    x0_range = 0.1
    u_center = [0.5]
    u_range = 0.5
    target_state = [0.5, 0.7]

    # 1. Generate Data
    logger.info("--- Stage 1: Generating Data ---")
    try:
        state_trajs, control_trajs, _ = adapter.generate_trajectories(
            10, 50, x0_center=x0_center, x0_range=x0_range, u_center=u_center, u_range=u_range
        )
    except Exception as e:
        logger.error(f"Failed to generate trajectories: {e}")
        traceback.print_exc()
        return

    # 2. Identify Koopman
    logger.info("--- Stage 2: Identification ---")
    try:
        A_k, B_k, C_k, phi_k, n_k = identify_koopman_model(adapter, state_trajs, control_trajs)
        koopman_model = (A_k, B_k, C_k, phi_k, n_k)
    except Exception as e:
        logger.error(f"ID failed: {e}")
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
            {'n_state_obs': adapter.n_state_obs, 'n_control': adapter.n_control, 'dt': adapter.dt},
            koopman_model, 10, theta_log, control_limits=control_limits, target_state=target_state
        )
    except Exception as e:
        logger.error(f"MPC init failed: {e}")
        traceback.print_exc()
        return

    # 4. Meta-Learning
    logger.info("--- Stage 4: Meta-Learning ---")
    x0_meta = np.array(x0_center).flatten() if x0_center is not None else np.random.uniform(-0.5,0.5,adapter.n_state_obs)
    logger.info(f"Target: {target_state}")

    try:
        theta_learned, meta_hist, theta_hist = learn_mpc_cost_params_pdp(
            adapter, mpc, 20, 5, 1e-4, x0_meta, target_state_meta=target_state
        )
        logger.info(f"Learned theta: {theta_learned}")
    except Exception as e:
        logger.error(f"Meta-learning failed: {e}")
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

    plot_results(adapter, hist_x, hist_u, title=f"{os.path.basename(script_dir)}_Final")
    logger.info("Done.")

if __name__ == "__main__":
    main()
