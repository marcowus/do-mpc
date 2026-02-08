import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../..')) # do_mpc
sys.path.append(os.path.join(script_dir, '..'))   # lib and adapter

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
    logger.info("Starting Prescribed-Time Koopman Control for CSTR")

    # 1. Setup System
    model = template_model()
    simulator = template_simulator(model)
    adapter = DoMPCAdapter(model, simulator)

    # Configuration
    # C_a, C_b, T_R, T_K
    x0_op = np.array([0.8, 0.5, 134.14, 130.0])

    # Start slightly away from OP to test regulation
    x0_perturbed = x0_op + np.array([0.0, 0.0, -2.0, -2.0])
    target_state = x0_op

    # 2. Generate Data
    logger.info("Generating Training Data...")
    x_center = x0_op.reshape(-1, 1)
    u_center = np.array([-10.0, 50.0]).reshape(-1, 1) # Q_dot, F

    X, U, _ = adapter.generate_trajectories(
        num_traj=20, traj_len=100,
        x0_center=x_center, x0_range=2.0,
        u_center=u_center, u_range=10.0
    )

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

    # 3. Train Koopman
    logger.info("Training Koopman Model...")
    km = EnhancedKoopman(n_obs=20, lambda_reg=1e-5)
    km.fit(X_train, Y_train, U_train, dt=adapter.dt)

    # 4. Setup Controller
    # Control T_R (idx 2) using T_K (idx 3). Input Q_dot (idx 0).
    Tp = 0.5 # hours? if dt=0.005h. 100 steps.

    controller = PrescribedTimeKoopmanBacksteppingController(
        target_state, Tp=Tp, sigma=(5.0, 5.0),
        dt=adapter.dt, alpha_switch=0.9, filter_alpha=0.8,
        state_indices=(2, 3), # TR, TK
        input_index=0, # Q_dot
        output_limits=(-5000, 5000) # Wide limits for Q_dot
    )

    Q_lqr = np.eye(4) * 10
    R_lqr = np.eye(2)
    controller.bind_koopman(km, Q=Q_lqr, R=R_lqr)

    # 5. Run Simulation
    logger.info("Running Closed-Loop Simulation...")
    x = x0_perturbed.copy()

    # Reset simulator
    adapter.simulator.x0 = x.reshape(-1, 1)
    adapter.simulator.reset_history()

    time_points = np.arange(0, Tp*1.5, adapter.dt)
    hist_x = [x]
    hist_u = []

    for t in time_points:
        u_val = controller.compute_control(x, t)

        # Assemble full input
        # Use steady state F from controller.u_sp
        u_full = controller.u_sp.copy()
        u_full[0] = u_val

        # Step
        x = adapter.dynamics_step(x, u_full)
        hist_x.append(x)
        hist_u.append(u_full)

    # Plotting
    hist_x = np.array(hist_x)
    hist_u = np.array(hist_u)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    # T_R
    ax[0].plot(time_points, hist_x[:-1, 2], label='T_R')
    ax[0].axhline(target_state[2], color='r', linestyle='--', label='Target')
    ax[0].set_ylabel('Reactor Temp [C]')
    ax[0].legend()

    # T_K
    ax[1].plot(time_points, hist_x[:-1, 3], label='T_K')
    ax[1].axhline(target_state[3], color='r', linestyle='--', label='Target')
    ax[1].set_ylabel('Jacket Temp [C]')
    ax[1].legend()

    # Input
    ax[2].plot(time_points, hist_u[:, 0], label='Q_dot')
    ax[2].set_ylabel('Cooling Rate [kW]')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig('cstr_prescribed_control.png')
    logger.info("Simulation complete. Plot saved to cstr_prescribed_control.png")

if __name__ == "__main__":
    main()
