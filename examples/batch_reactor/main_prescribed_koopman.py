
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, r'../..'))
sys.path.append(os.path.join(script_dir, r'..'))

import do_mpc
from prescribed_time_koopman_lib import EnhancedKoopman, PrescribedTimeKoopmanBacksteppingController, calculate_performance_metrics
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
    logger.info("Starting Prescribed-Time Koopman Control")

    try:
        model = template_model()
        simulator = template_simulator(model)
        adapter = DoMPCAdapter(model, simulator)
    except Exception as e:
        logger.error(f"Failed to setup system: {e}")
        return

    x0_op = np.array([1.0, 0.5, 0.0, 120.0])
    target_state = np.array([1.0, 0.5, 0.0, 120.0])
    x0_perturbed = x0_op + np.array([0.0, 0.1, 0.0, 0.0])

    # Generate Data
    logger.info("Generating Training Data...")
    x_center = x0_op.reshape(-1, 1)
    u_center = np.array([0.5]).reshape(-1, 1)

    try:
        X, U, _ = adapter.generate_trajectories(
            num_traj=20, traj_len=100,
            x0_center=x_center, x0_range=0.1,
            u_center=u_center, u_range=0.5
        )
    except Exception as e:
        logger.error(f"Failed to generate trajectories: {e}")
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
    Tp = 1.0
    controller = PrescribedTimeKoopmanBacksteppingController(
        target_state, Tp=Tp, sigma=(5.0, 5.0),
        dt=adapter.dt, alpha_switch=0.9, filter_alpha=0.8,
        state_indices=(2, 1),
        input_index=0,
        output_limits=(0, 5)
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
        if len(u_full) > 0:
             u_full[0] = u_val
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
        ax[0].plot(time_points, hist_x[:-1, i], label=f'x{i}')
    ax[0].set_title('States')
    ax[0].legend()

    for i in range(hist_u.shape[1]):
        ax[1].plot(time_points, hist_u[:, i], label=f'u{i}')
    ax[1].set_title('Inputs')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('prescribed_control_results.png')
    logger.info("Done. Plot saved to prescribed_control_results.png")

if __name__ == "__main__":
    main()
