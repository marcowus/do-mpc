import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from scipy.linalg import solve_continuous_are

# ===================== Data Generator =====================
class DataGenerator:
    def __init__(self, system_dynamics, dt=0.1):
        self.system = system_dynamics
        self.dt = dt

    def generate_excitation(self, t, phase=0):
        # Excitation signal suitable for process control
        return (
            0.02 * np.sin(0.1 * t + phase) +
            0.01 * np.sin(0.8 * t) +
            0.03 * (t % 40 < 20) +
            0.005 * np.random.randn()
        )

    def generate_data(self, num_trajectories=50, steps_per_traj=800, u_min=0, u_max=0.1, x_bounds=None):
        X, Y, U = [], [], []
        for _ in range(num_trajectories):
            # Initial states
            if x_bounds:
                x = np.array([np.random.uniform(low, high) for low, high in x_bounds])
            else:
                x = np.random.rand(2) # Default fallback

            phase = np.random.uniform(0, 2 * np.pi)
            for k in range(steps_per_traj):
                t = k * self.dt
                # Generate scalar input
                excitation = self.generate_excitation(t, phase)
                # Scale excitation to [u_min, u_max] roughly
                u_base = (u_max + u_min) / 2
                u_amp = (u_max - u_min) / 2
                u_in = np.clip(u_base + excitation * (u_amp/0.05), u_min, u_max)

                dx = self.system.dynamics(t, x, u_in)
                x_next = x + dx * self.dt
                X.append(x.copy()); Y.append(x_next.copy()); U.append(u_in)
                x = x_next
        return np.array(X), np.array(Y), np.array(U)

# ===================== Koopman Model =====================
class EnhancedKoopman(BaseEstimator):
    def __init__(self, n_obs=10, lambda_reg=0.01):
        self.n_obs = n_obs
        self.lambda_reg = lambda_reg
        self.A = None # Continuous-time state matrix
        self.B = None # Continuous-time input matrix
        self.dt = None

    def lift(self, x):
        # Assumes x has at least 2 elements.
        # Adapting to potentially more states by taking first 2 principal ones or custom logic
        # For this paradigm, we stick to the user's specific lifting functions for T_tube, T_shell
        # but try to be generic if x has different meaning.
        # Ideally, this should be customizable.
        # For now, we use the user's lift function which is designed for 2 states.
        x = np.asarray(x).flatten()
        if len(x) < 2:
            return np.array([x[0], x[0]**2, 1.0]) # Minimal fallback

        T_tube, T_shell = x[:2]
        # User provided features:
        return np.array([
            T_tube,
            T_shell,
            T_tube**2,
            T_shell**2,
            T_tube * T_shell,
            (T_shell - T_tube), # Heat transfer driving force
            np.sin(T_tube / 50),
            np.cos(T_shell / 50),
            (T_shell - T_tube)**2, # An additional nonlinear term
            1.0                      # Constant term
        ])[:self.n_obs]

    def fit(self, X, Y, U, dt):
        self.dt = dt
        U = np.asarray(U).reshape(-1, 1)
        Psi_X = np.array([self.lift(x) for x in X])
        Psi_Y = np.array([self.lift(y) for y in Y])

        Omega = np.hstack([Psi_X, U])
        try:
            # Use Ridge Regression to solve for discrete-time Koopman operators Ad, Bd
            AB_d_T = np.linalg.pinv(Omega.T @ Omega + self.lambda_reg * np.eye(Omega.shape[1])) @ Omega.T @ Psi_Y
            AB_d = AB_d_T.T
        except np.linalg.LinAlgError:
            print("Warning: Matrix inversion failed. Check your data.")
            return self

        Ad = AB_d[:, :-1]
        Bd = AB_d[:, -1]

        # Convert to continuous-time matrices A, B
        self.A = (Ad - np.eye(self.n_obs)) / self.dt
        self.B = Bd / self.dt
        return self

    def get_linearized_system(self, x0, u0):
        # Project back to original state space (first 2 states)
        C = np.zeros((2, self.n_obs)); C[0, 0] = 1; C[1, 1] = 1
        psi_x0 = self.lift(x0)

        eps_x = 1e-4; eps_u = 1e-4
        A_lin = np.zeros((2, 2))
        f0 = C @ (self.A @ psi_x0 + self.B * u0)

        for i in range(2):
            x_plus = x0.copy(); x_plus[i] += eps_x
            f_plus = C @ (self.A @ self.lift(x_plus) + self.B * u0)
            A_lin[:, i] = (f_plus - f0) / eps_x

        f_plus_u = C @ (self.A @ psi_x0 + self.B * (u0 + eps_u))
        B_lin = ((f_plus_u - f0) / eps_u).reshape(-1, 1)

        return A_lin, B_lin

# ===================== Prescribed-Time Koopman Backstepping Controller =====================
class PrescribedTimeKoopmanBacksteppingController:
    def __init__(self, target_state, Tp, sigma, dt, alpha_switch, filter_alpha, output_limits=(0, 0.1)):
        self.target = np.asarray(target_state)
        self.T_tube_d, self.T_shell_d = self.target[:2]
        self.Tp = Tp
        self.sigma = np.array(sigma)
        self.dt = dt
        self.alpha_switch = alpha_switch
        self.filter_alpha = filter_alpha
        self.output_min, self.output_max = output_limits

        self.koopman = None
        self.u_filtered = 0.0
        self.alpha1_prev = self.T_shell_d

        self.K_lqr = None
        self.u_sp = 0.0

    def bind_koopman(self, koopman_model, Q=np.diag([10, 1]), R=np.array([[100]])):
        self.koopman = koopman_model
        psi_target = self.koopman.lift(self.target)
        try:
            self.u_sp, _, _, _ = np.linalg.lstsq(-self.koopman.B.reshape(-1, 1), self.koopman.A @ psi_target, rcond=None)
            self.u_sp = self.u_sp[0]
        except np.linalg.LinAlgError:
            self.u_sp = 0.02 # Fallback value
            print("Warning: Failed to solve for steady-state input. Using default.")

        A_lin, B_lin = self.koopman.get_linearized_system(self.target, self.u_sp)
        try:
            P = solve_continuous_are(A_lin, B_lin, Q, R)
            self.K_lqr = np.linalg.inv(R) @ B_lin.T @ P
        except np.linalg.LinAlgError:
            print("Warning: LQR solution failed. Using zero gain.")
            self.K_lqr = np.zeros((1, 2))

    def compute_control(self, x, t):
        if self.koopman is None: raise RuntimeError("Koopman model not bound.")

        # Switch to LQR near the end of prescribed time
        if t >= self.alpha_switch * self.Tp:
            u_raw = self.linear_lqr_control(x)
        else:
            u_raw = self.time_varying_control(x, t)

        self.u_filtered = self.filter_alpha * self.u_filtered + (1 - self.filter_alpha) * u_raw
        return np.clip(self.u_filtered, self.output_min, self.output_max)

    def time_varying_control(self, x, t):
        T_tube, T_shell = x[:2]
        mu_t = 1.0 / max(self.Tp - t, 1e-4)

        C = np.zeros((2, self.koopman.n_obs)); C[0, 0] = 1; C[1, 1] = 1

        # Backstepping Design: u -> T_shell -> T_tube
        # z1: Error of target variable
        z1 = T_tube - self.T_tube_d

        # alpha1: Virtual control for T_shell
        alpha1 = self.T_shell_d - self.sigma[0] * z1
        alpha1_dot = (alpha1 - self.alpha1_prev) / self.dt
        self.alpha1_prev = alpha1

        # z2: Error of virtual control
        z2 = T_shell - alpha1

        # Estimate dynamics of T_shell (f2) using Koopman
        f2_hat = C[1,:] @ (self.koopman.A @ self.koopman.lift(x))
        B2_hat = C[1,:] @ self.koopman.B

        if abs(B2_hat) < 1e-5: B2_hat = np.sign(B2_hat) * 1e-5 if B2_hat != 0 else 1e-5

        # Control law to stabilize z2 -> 0 in prescribed time
        u = (1/B2_hat) * (alpha1_dot - f2_hat - self.sigma[1] * mu_t * z2 - z1)

        return u

    def linear_lqr_control(self, x):
        e = x[:2] - self.target[:2]
        u_adjustment = -self.K_lqr @ e
        return self.u_sp + u_adjustment[0]

# ===================== PID Controller =====================
class PIDController:
    """A simple PID controller implementation"""
    def __init__(self, Kp, Ki, Kd, target, dt, output_limits=(0, 0.1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.dt = dt
        self.output_min, self.output_max = output_limits

        self.integral = 0
        self.prev_error = 0

    def compute(self, measurement):
        error = self.target - measurement

        # Proportional term
        P = self.Kp * error

        # Integral term (with anti-windup)
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative

        # Compute and clamp output
        output = P + I + D
        clamped_output = np.clip(output, self.output_min, self.output_max)

        # Anti-windup for integral term
        if output != clamped_output:
            self.integral -= error * self.dt

        self.prev_error = error
        return clamped_output
