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
class LotkaVolterraWrapper:
    def __init__(self):
        self.model = template_model()
        self.simulator = template_simulator(self.model)
        self.simulator.setup()
        self.simulator.x0 = np.array([0.5, 0.7]) # Default initial state

    def dynamics(self, t, x, u):
        # Reset simulator to current state x
        self.simulator.x0 = x
        self.simulator.u0 = np.array([u])

        # We need dx/dt, but do-mpc simulator gives x_next.
        # We can approximate dx = (x_next - x) / dt if using discrete simulator,
        # or use model equations directly if available.
        # Here we use the model equations directly for better accuracy in DataGeneration.

        # Extract parameters
        c0 = 0.4
        c1 = 0.2

        x0 = x[0] # Prey
        x1 = x[1] # Predator

        # Model equations from template_model.py:
        # dx0 = x0 - x0 * x1 - c0 * x0 * u
        # dx1 = -x1 + x0 * x1 - c1 * x1 * u

        dx0 = x0 - x0 * x1 - c0 * x0 * u
        dx1 = -x1 + x0 * x1 - c1 * x1 * u

        return np.array([dx0, dx1])

def main():
    dt = 0.1
    system = LotkaVolterraWrapper()

    print("Generating training data...")
    dg = DataGenerator(system, dt=dt)
    # Generate data with random initial conditions and inputs
    X_train, Y_train, U_train = dg.generate_data(
        num_trajectories=50,
        steps_per_traj=200,
        u_min=0.0,
        u_max=1.0,
        x_bounds=[(0.1, 2.0), (0.1, 2.0)]
    )

    print("Training Koopman model...")
    koopman_model = EnhancedKoopman(n_obs=10, lambda_reg=0.01)
    koopman_model.fit(X_train, Y_train, U_train, dt)

    # Define Control Task
    # Goal: Drive Prey (x0) to a target value.
    # The backstepping controller assumes u -> x1 -> x0 (or similar).
    # In LV: u affects x0 directly (-c0*x0*u) AND x1 directly (-c1*x1*u).
    # This is NOT strict feedback form.
    # However, let's try to control x0 to 1.0, assuming we can find a steady state for x1.
    # Steady state relations (dx=0):
    # 0 = x0 - x0*x1 - c0*x0*u => 1 - x1 - c0*u = 0 (if x0 != 0) => x1 = 1 - c0*u
    # 0 = -x1 + x0*x1 - c1*x1*u => -1 + x0 - c1*u = 0 (if x1 != 0) => x0 = 1 + c1*u

    # If target x0 = 1.2:
    # 1.2 = 1 + 0.2*u => 0.2*u = 0.2 => u = 1.0
    # Then x1 = 1 - 0.4*1.0 = 0.6

    T_target_x0 = 1.2
    u_steady = (T_target_x0 - 1) / 0.2
    T_target_x1 = 1 - 0.4 * u_steady

    target_state = np.array([T_target_x0, T_target_x1])
    print(f"Target: x0={T_target_x0}, x1={T_target_x1}, u={u_steady}")

    Tp_sim = 20.0
    simulation_time = 25.0

    # Note: The controller assumes u controls x1 (shell) which controls x0 (tube).
    # In LV, u affects x0 directly. This might confuse the backstepping logic which assumes x1 is the virtual control for x0.
    # But let's see if the Koopman learning can capture the dynamics well enough for the controller to work,
    # or if the structure mismatch causes failure.
    # The controller logic: u = (1/B2) * ( ... ) to drive z2 (error in x1).
    # If B2 (influence of u on x1) is significant, it might work to track x1 trajectory.
    # But does tracking x1 drive x0 to target?
    # In the controller, alpha1 (desired x1) is computed to drive z1 (error in x0) to zero.
    # alpha1 = x1_d - sigma * z1.
    # So it tries to move x1 to a value that fixes x0 error.
    # Dynamics of x0: dx0 = x0(1 - x1 - c0*u). If u is roughly constant, x0 grows if x1 < 1-c0*u.
    # So decreasing x1 increases x0. The controller should handle this sign relationship.

    controller = PrescribedTimeKoopmanBacksteppingController(
        target_state, Tp=Tp_sim, sigma=(1.0, 1.0),
        dt=dt, alpha_switch=0.9, filter_alpha=0.8,
        output_limits=(0.0, 1.5)
    )
    controller.bind_koopman(koopman_model)

    initial_states = [np.array([0.5, 0.5]), np.array([1.5, 1.0])]
    logs = []

    print("Running Koopman Controller simulations...")
    for init_state in initial_states:
        log_data = {'time': np.arange(0, simulation_time, dt), 'states': [], 'control': []}
        x_real = init_state.copy()
        for t_step in log_data['time']:
            u = controller.compute_control(x_real, t_step)
            dx = system.dynamics(t_step, x_real, u)
            x_real += dx * dt
            log_data['states'].append(x_real.copy())
            log_data['control'].append(u)
        log_data['states'] = np.array(log_data['states'])
        logs.append(log_data)

    # PID Comparison
    print("Running PID Controller simulation...")
    # PID controlling x0 directly
    pid = PIDController(Kp=2.0, Ki=0.5, Kd=0.1, target=T_target_x0, dt=dt, output_limits=(0.0, 1.5))
    pid_log = {'time': np.arange(0, simulation_time, dt), 'states': [], 'control': []}
    x_real = initial_states[0].copy()
    for t_step in pid_log['time']:
        u = pid.compute(x_real[0])
        dx = system.dynamics(t_step, x_real, u)
        x_real += dx * dt
        pid_log['states'].append(x_real.copy())
        pid_log['control'].append(u)
    pid_log['states'] = np.array(pid_log['states'])

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i, log in enumerate(logs):
        axs[0].plot(log['time'], log['states'][:, 0], label=f'Koopman Sim {i+1}')
        axs[1].plot(log['time'], log['states'][:, 1], label=f'Koopman Sim {i+1}')
        axs[2].plot(log['time'], log['control'], label=f'Koopman Sim {i+1}')

    axs[0].plot(pid_log['time'], pid_log['states'][:, 0], 'r--', label='PID')
    axs[1].plot(pid_log['time'], pid_log['states'][:, 1], 'r--', label='PID')
    axs[2].plot(pid_log['time'], pid_log['control'], 'r--', label='PID')

    axs[0].axhline(T_target_x0, color='k', linestyle=':', label='Target x0')
    axs[1].axhline(T_target_x1, color='k', linestyle=':', label='Steady x1')

    axs[0].set_ylabel('Prey (x0)')
    axs[1].set_ylabel('Predator (x1)')
    axs[2].set_ylabel('Input (u)')
    axs[2].set_xlabel('Time')
    axs[0].legend()

    plt.tight_layout()
    plt.savefig('test_koopman_results.png')
    print("Results saved to test_koopman_results.png")

if __name__ == "__main__":
    main()
