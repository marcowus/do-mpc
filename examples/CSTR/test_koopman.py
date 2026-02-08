import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add path to access examples
sys.path.append(os.path.join('..'))
from Koopman_Paradigm import DataGenerator, EnhancedKoopman, PrescribedTimeKoopmanBacksteppingController, PIDController

# Add path to access do-mpc
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from template_model import template_model
from template_simulator import template_simulator

# Wrapper for do-mpc simulator to match DataGenerator interface
class CSTRWrapper:
    def __init__(self):
        self.model = template_model()
        self.simulator = template_simulator(self.model)
        self.simulator.setup()

        # Initial parameters
        self.simulator.x0 = np.array([0.8, 0.5, 134.14, 130.0]).reshape(-1, 1)
        self.F_default = 100.0 # Fixed flow rate for this experiment

    def dynamics(self, t, x, u):
        # x = [C_a, C_b, T_R, T_K]
        # u = Q_dot (Heat removal/addition)

        # Reset simulator
        self.simulator.x0 = x.reshape(-1, 1)
        # Input u0 has shape (2,1) for [F, Q_dot]
        self.simulator.u0 = np.array([self.F_default, u]).reshape(-1, 1)

        # Using model equations directly for continuous dynamics approximation
        # Need to re-implement RHS here or fetch from model if possible
        # Fetching parameters from model definition (hardcoded here for simplicity as extracting from CasADi structure is complex at runtime)

        C_a, C_b, T_R, T_K = x
        F = self.F_default
        Q_dot = u

        # Parameters from template_model.py
        K0_ab = 1.287e12
        K0_bc = 1.287e12
        K0_ad = 9.043e9
        # R_gas = 8.3144621e-3
        E_A_ab = 9758.3
        E_A_bc = 9758.3
        E_A_ad = 8560.0
        H_R_ab = 4.2
        H_R_bc = -11.0
        H_R_ad = -41.85
        Rou = 0.9342
        Cp = 3.01
        Cp_k = 2.0
        A_R = 0.215
        V_R = 10.01
        m_k = 5.0
        T_in = 130.0
        K_w = 4032.0
        C_A0 = 5.1
        alpha = 1.0
        beta = 1.0

        T_dif = T_R - T_K

        K_1 = beta * K0_ab * np.exp((-E_A_ab)/((T_R+273.15)))
        K_2 =  K0_bc * np.exp((-E_A_bc)/((T_R+273.15)))
        K_3 = K0_ad * np.exp((-alpha*E_A_ad)/((T_R+273.15)))

        dC_a = F*(C_A0 - C_a) - K_1*C_a - K_3*(C_a**2)
        dC_b = -F*C_b + K_1*C_a - K_2*C_b
        dT_R = ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R))
        dT_K = (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k)

        return np.array([dC_a, dC_b, dT_R, dT_K])

def main():
    dt = 0.005 # Small dt for fast chemical dynamics
    system = CSTRWrapper()

    print("Generating training data...")
    dg = DataGenerator(system, dt=dt)
    # Generate data
    # States: C_a, C_b, T_R, T_K
    # We focus on T_R (Reactor Temp) and T_K (Cooling Temp) for the backstepping control.
    # Q_dot -> T_K -> T_R.
    # So we need to lift x to include these.
    # We will pass full state to fit, but Koopman might focus on relevant ones.

    X_train, Y_train, U_train = dg.generate_data(
        num_trajectories=30,
        steps_per_traj=200,
        u_min=-10.0,
        u_max=10.0,
        x_bounds=[(0.1, 2.0), (0.1, 2.0), (100, 150), (100, 150)]
    )

    print("Training Koopman model...")
    # Custom lifting for 4 states?
    # The default EnhancedKoopman expects 2 states in lift().
    # We need to adapt EnhancedKoopman or wrap the state to only expose T_R (idx 2) and T_K (idx 3).
    # Let's create a partial view of the system where x = [T_R, T_K].
    # But T_R dynamics depend on C_a, C_b.
    # If we ignore concentration dynamics (assuming slow or decoupled), it's an approximation.
    # Alternatively, we can subclass EnhancedKoopman to handle 4 states.

    # For this demonstration of "applying the paradigm", I will adapt the input data to only show [T_R, T_K] to the Koopman learner,
    # effectively learning a reduced order model.

    X_train_reduced = X_train[:, [2, 3]] # T_R, T_K
    Y_train_reduced = Y_train[:, [2, 3]]

    koopman_model = EnhancedKoopman(n_obs=10, lambda_reg=0.01)
    koopman_model.fit(X_train_reduced, Y_train_reduced, U_train, dt)

    # Control Task: Control T_R using Q_dot.
    # Chain: Q_dot -> T_K -> T_R.
    # Target: T_R = 120.0
    # Steady state estimation is hard without full model.
    # Let's pick a feasible target.

    T_R_target = 120.0
    # Rough estimate of steady T_K for T_R=120
    # At steady state (ignoring reaction heat for approx):
    # F(Tin - TR) + UA/V (TK - TR) = 0
    # 100(130 - 120) + (4032*0.215)/(0.9342*3.01*10.01) * (TK - 120) = 0
    # 1000 + 30.8 * (TK - 120) = 0 => TK - 120 = -32 => TK = 88
    T_K_target = 88.0
    u_steady = -5.0 # Guess

    target_state = np.array([T_R_target, T_K_target])

    Tp_sim = 1.0 # Fast response needed
    simulation_time = 1.5

    controller = PrescribedTimeKoopmanBacksteppingController(
        target_state, Tp=Tp_sim, sigma=(5.0, 5.0),
        dt=dt, alpha_switch=0.9, filter_alpha=0.5,
        output_limits=(-50.0, 50.0)
    )
    controller.bind_koopman(koopman_model)

    initial_states = [
        np.array([0.8, 0.5, 135.0, 130.0]),
        np.array([0.8, 0.5, 110.0, 100.0])
    ]
    logs = []

    print("Running Koopman Controller simulations...")
    for init_state in initial_states:
        log_data = {'time': np.arange(0, simulation_time, dt), 'states': [], 'control': []}
        x_real = init_state.copy()
        for t_step in log_data['time']:
            # Controller only sees [T_R, T_K]
            x_view = x_real[[2, 3]]
            u = controller.compute_control(x_view, t_step)

            dx = system.dynamics(t_step, x_real, u)
            x_real += dx * dt
            log_data['states'].append(x_real.copy())
            log_data['control'].append(u)
        log_data['states'] = np.array(log_data['states'])
        logs.append(log_data)

    # PID Comparison on T_R
    print("Running PID Controller simulation...")
    pid = PIDController(Kp=-20.0, Ki=-50.0, Kd=-1.0, target=T_R_target, dt=dt, output_limits=(-50.0, 50.0))
    pid_log = {'time': np.arange(0, simulation_time, dt), 'states': [], 'control': []}
    x_real = initial_states[0].copy()
    for t_step in pid_log['time']:
        u = pid.compute(x_real[2])
        dx = system.dynamics(t_step, x_real, u)
        x_real += dx * dt
        pid_log['states'].append(x_real.copy())
        pid_log['control'].append(u)
    pid_log['states'] = np.array(pid_log['states'])

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i, log in enumerate(logs):
        axs[0].plot(log['time'], log['states'][:, 2], label=f'Koopman Sim {i+1}')
        axs[1].plot(log['time'], log['states'][:, 3], label=f'Koopman Sim {i+1}')
        axs[2].plot(log['time'], log['control'], label=f'Koopman Sim {i+1}')

    axs[0].plot(pid_log['time'], pid_log['states'][:, 2], 'r--', label='PID')
    axs[1].plot(pid_log['time'], pid_log['states'][:, 3], 'r--', label='PID')
    axs[2].plot(pid_log['time'], pid_log['control'], 'r--', label='PID')

    axs[0].axhline(T_R_target, color='k', linestyle=':', label='Target T_R')
    axs[1].axhline(T_K_target, color='k', linestyle=':', label='Steady T_K')

    axs[0].set_ylabel('Reactor Temp (T_R)')
    axs[1].set_ylabel('Jacket Temp (T_K)')
    axs[2].set_ylabel('Heat Input (Q_dot)')
    axs[2].set_xlabel('Time')
    axs[0].legend()

    plt.tight_layout()
    plt.savefig('test_koopman_cstr_results.png')
    print("Results saved to test_koopman_cstr_results.png")

if __name__ == "__main__":
    main()
