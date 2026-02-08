import os
import sys

def get_prescribed_config(dir_path):
    config = {'run': False, 'model_init': 'model = template_model()'}

    if "CSTR" in dir_path and "approximate" not in dir_path and "lqr" not in dir_path:
        config['run'] = True
        config['x0_op'] = "np.array([0.8, 0.5, 134.14, 130.0])"
        config['target_state'] = "np.array([0.8, 0.5, 134.14, 130.0])"
        config['state_indices'] = "(2, 3)" # TR, TK
        config['input_index'] = "0" # Q_dot
        config['u_center'] = "np.array([-10.0, 50.0]).reshape(-1, 1)"
        config['u_range'] = "10.0"
        config['x0_range'] = "2.0"
        config['output_limits'] = "(-5000, 5000)"
        config['Tp'] = "0.5"
        config['x0_perturb'] = "np.array([0.0, 0.0, -2.0, -2.0])"

    elif "triple_tank" in dir_path:
        config['run'] = True
        config['x0_op'] = "np.array([10.0, 10.0, 10.0])"
        config['target_state'] = "np.array([10.0, 10.0, 10.0])"
        # Control x3 (idx 2) via x1 (idx 0). Input u1 (idx 0).
        config['state_indices'] = "(2, 0)"
        config['input_index'] = "0"
        config['u_center'] = "np.array([0.5, 0.5]).reshape(-1, 1)"
        config['u_range'] = "0.5"
        config['x0_range'] = "2.0"
        config['output_limits'] = "(-1, 2)"
        config['Tp'] = "50.0" # Discrete time steps? Ts=1. So 50 steps.
        config['x0_perturb'] = "np.array([2.0, 0.0, -2.0])"

    elif "batch_reactor" in dir_path and "lqr" not in dir_path and "differentiator" not in dir_path:
        config['run'] = True
        config['x0_op'] = "np.array([1.0, 0.5, 0.0, 120.0])"
        config['target_state'] = "np.array([1.0, 0.5, 0.0, 120.0])" # Stay at initial?
        # Control P (idx 2) via S (idx 1).
        config['state_indices'] = "(2, 1)"
        config['input_index'] = "0"
        config['u_center'] = "np.array([0.5]).reshape(-1, 1)"
        config['u_range'] = "0.5"
        config['x0_range'] = "0.1"
        config['output_limits'] = "(0, 5)"
        config['Tp'] = "1.0"
        config['x0_perturb'] = "np.array([0.0, 0.1, 0.0, 0.0])"

    return config

def generate_main_prescribed(dir_path):
    config = get_prescribed_config(dir_path)
    if not config['run']:
        return

    print(f"Generating main_prescribed_koopman.py in {dir_path}")

    abs_dir = os.path.abspath(dir_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    rel_path_to_root = os.path.relpath(repo_root, abs_dir)
    rel_path_to_examples = os.path.relpath(os.path.dirname(__file__), abs_dir)

    content = f"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, r'{rel_path_to_root}'))
sys.path.append(os.path.join(script_dir, r'{rel_path_to_examples}'))

import do_mpc
from prescribed_time_koopman_lib import EnhancedKoopman, PrescribedTimeKoopmanBacksteppingController, calculate_performance_metrics
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
    logger.info("Starting Prescribed-Time Koopman Control")

    try:
        {config['model_init']}
        simulator = template_simulator(model)
        adapter = DoMPCAdapter(model, simulator)
    except Exception as e:
        logger.error(f"Failed to setup system: {{e}}")
        return

    x0_op = {config['x0_op']}
    target_state = {config['target_state']}
    x0_perturbed = x0_op + {config['x0_perturb']}

    # Generate Data
    logger.info("Generating Training Data...")
    x_center = x0_op.reshape(-1, 1)
    u_center = {config['u_center']}

    try:
        X, U, _ = adapter.generate_trajectories(
            num_traj=20, traj_len=100,
            x0_center=x_center, x0_range={config['x0_range']},
            u_center=u_center, u_range={config['u_range']}
        )
    except Exception as e:
        logger.error(f"Failed to generate trajectories: {{e}}")
        return

    X_train, Y_train, U_train = [], [], []
    for i in range(len(X)):
        states = X[i]
        controls = U[i]
        for k in range(len(controls)):
            X_train.append(states[k])
            Y_train.append(states[k+1])
            U_train.append(controls[k])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    U_train = np.array(U_train)

    # Train Koopman
    logger.info("Training Koopman Model...")
    km = EnhancedKoopman(n_obs=20, lambda_reg=1e-5)
    km.fit(X_train, Y_train, U_train, dt=adapter.dt)

    # Controller
    Tp = {config['Tp']}
    controller = PrescribedTimeKoopmanBacksteppingController(
        target_state, Tp=Tp, sigma=(5.0, 5.0),
        dt=adapter.dt, alpha_switch=0.9, filter_alpha=0.8,
        state_indices={config['state_indices']},
        input_index={config['input_index']},
        output_limits={config['output_limits']}
    )

    n_x = adapter.n_state_obs
    n_u = adapter.n_control
    controller.bind_koopman(km, Q=np.eye(n_x)*10, R=np.eye(n_u))

    # Simulation
    logger.info("Running Closed-Loop Simulation...")
    x = x0_perturbed.copy()
    adapter.simulator.x0 = x.reshape(-1, 1)
    adapter.simulator.reset_history()

    time_points = np.arange(0, Tp*1.5, adapter.dt)
    hist_x = [x]
    hist_u = []

    for t in time_points:
        u_val = controller.compute_control(x, t)

        # Assemble full input
        u_full = controller.u_sp.copy()
        if len(u_full) > {config['input_index']}:
             u_full[{config['input_index']}] = u_val
        else:
             u_full = np.array([u_val]) # Fallback

        x = adapter.dynamics_step(x, u_full)
        hist_x.append(x)
        hist_u.append(u_full)

    hist_x = np.array(hist_x)
    hist_u = np.array(hist_u)

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    for i in range(hist_x.shape[1]):
        ax[0].plot(time_points, hist_x[:-1, i], label=f'x{{i}}')
    ax[0].set_title('States')
    ax[0].legend()

    for i in range(hist_u.shape[1]):
        ax[1].plot(time_points, hist_u[:, i], label=f'u{{i}}')
    ax[1].set_title('Inputs')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('prescribed_control_results.png')
    logger.info("Done. Plot saved to prescribed_control_results.png")

if __name__ == "__main__":
    main()
"""

    output_path = os.path.join(dir_path, "main_prescribed_koopman.py")
    with open(output_path, "w") as f:
        f.write(content)

def find_and_deploy():
    root_examples = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(root_examples):
        if "template_model.py" in files:
            generate_main_prescribed(root)

if __name__ == "__main__":
    find_and_deploy()
