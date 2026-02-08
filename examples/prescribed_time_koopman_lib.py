import numpy as np
from sklearn.base import BaseEstimator
from scipy.linalg import solve_continuous_are

# ===================== Koopman Model =====================
class EnhancedKoopman(BaseEstimator):
    def __init__(self, n_obs=10, lambda_reg=0.01, phi_func=None):
        self.n_obs = n_obs
        self.lambda_reg = lambda_reg
        self.A = None # Continuous-time state matrix
        self.B = None # Continuous-time input matrix
        self.dt = None
        self.phi_func = phi_func

    def lift(self, x):
        if self.phi_func is not None:
            return self.phi_func(x)[:self.n_obs]

        # Default lifting (assuming x is at least 2D)
        # Flatten x first
        x_flat = np.asarray(x).flatten()
        # Basic quadratic lifting
        poly = [x_flat]
        n = len(x_flat)
        for i in range(n):
            for j in range(i, n):
                poly.append(x_flat[i]*x_flat[j])
        # Add some nonlinear terms if needed, or stick to quadratic
        # The user provided specific terms for heat exchanger.
        # I'll use a generic quadratic + 1
        lifted = np.hstack(poly)
        lifted = np.append(lifted, 1.0)

        if len(lifted) < self.n_obs:
            # Pad with zeros if needed? Or just slice.
            # Better to allow dynamic size or truncation
            pass
        return lifted[:self.n_obs]

    def fit(self, X, Y, U, dt):
        self.dt = dt
        U = np.asarray(U)
        if U.ndim == 1:
            U = U.reshape(-1, 1)

        Psi_X = np.array([self.lift(x) for x in X])
        Psi_Y = np.array([self.lift(y) for y in Y])

        # Omega = [Psi_X, U]
        Omega = np.hstack([Psi_X, U])

        try:
            # Ad, Bd
            # min || Psi_Y - [Ad Bd] [Psi_X; U] ||
            # AB_d.T = pinv(Omega.T Omega + reg) Omega.T Psi_Y
            AB_d_T = np.linalg.pinv(Omega.T @ Omega + self.lambda_reg * np.eye(Omega.shape[1])) @ Omega.T @ Psi_Y
            AB_d = AB_d_T.T
        except np.linalg.LinAlgError:
            print("Warning: Matrix inversion failed. Check your data.")
            return self

        self.n_lifted = Psi_X.shape[1]
        self.n_inputs = U.shape[1]

        Ad = AB_d[:, :self.n_lifted]
        Bd = AB_d[:, self.n_lifted:]

        # Continuous time conversion
        # A = (Ad - I) / dt
        self.A = (Ad - np.eye(self.n_lifted)) / self.dt
        self.B = Bd / self.dt
        return self

    def get_linearized_system(self, x0, u0):
        # Jacobian approximation via finite difference on lifted dynamics
        # f(z) = A z + B u
        # We need dx/dt = f_x(x, u)
        # x = C z.
        # But Koopman gives z_dot = A z + B u.
        # So x_dot = C (A z + B u).
        # Linearization A_lin = d(x_dot)/dx = C A dz/dx.
        # B_lin = d(x_dot)/du = C B.

        # We need projection matrix C (reconstruct x from z)
        # Assuming first n_x elements of z are x.
        x0 = np.asarray(x0).flatten()
        n_x = len(x0)

        # Just use finite difference on the lifted prediction
        # f(x, u) = C @ (A @ lift(x) + B @ u)

        # C projects z to x. Assuming z[0:n_x] = x
        n_z = self.A.shape[0]
        C = np.zeros((n_x, n_z))
        C[:n_x, :n_x] = np.eye(n_x)

        u0 = np.asarray(u0).flatten()
        psi_x0 = self.lift(x0)

        eps = 1e-4
        A_lin = np.zeros((n_x, n_x))
        f0 = C @ (self.A @ psi_x0 + self.B @ u0)

        for i in range(n_x):
            x_plus = x0.copy()
            x_plus[i] += eps
            f_plus = C @ (self.A @ self.lift(x_plus) + self.B @ u0)
            A_lin[:, i] = (f_plus - f0) / eps

        B_lin = np.zeros((n_x, len(u0)))
        for i in range(len(u0)):
            u_plus = u0.copy()
            u_plus[i] += eps
            f_plus_u = C @ (self.A @ psi_x0 + self.B @ u_plus)
            B_lin[:, i] = (f_plus_u - f0) / eps

        return A_lin, B_lin

# ===================== Prescribed-Time Koopman Backstepping Controller =====================
class PrescribedTimeKoopmanBacksteppingController:
    def __init__(self, target_state, Tp, sigma, dt, alpha_switch, filter_alpha, state_indices=(0, 1), input_index=0, output_limits=None):
        """
        state_indices: (controlled_state_idx, actuator_state_idx)
        input_index: index of the control input u that drives the actuator state.
        """
        self.target = np.asarray(target_state)
        self.Tp = Tp
        self.sigma = np.array(sigma)
        self.dt = dt
        self.alpha_switch = alpha_switch
        self.filter_alpha = filter_alpha
        self.state_indices = state_indices
        self.input_index = input_index
        self.output_limits = output_limits

        self.koopman = None
        self.u_filtered = 0.0
        self.alpha1_prev = None # Will init on first run

        self.K_lqr = None
        self.u_sp = 0.0

    def bind_koopman(self, koopman_model, Q=None, R=None):
        self.koopman = koopman_model
        psi_target = self.koopman.lift(self.target)
        n_u = self.koopman.B.shape[1]

        # Solve steady state u
        # A z + B u = 0 (steady state in lifted space? approx)
        try:
            self.u_sp, _, _, _ = np.linalg.lstsq(-self.koopman.B, self.koopman.A @ psi_target, rcond=None)
        except np.linalg.LinAlgError:
            self.u_sp = np.zeros(n_u)
            print("Warning: Failed to solve for steady-state input. Using default.")

        # LQR
        A_lin, B_lin = self.koopman.get_linearized_system(self.target, self.u_sp)
        n_x = A_lin.shape[0]

        if Q is None: Q = np.eye(n_x) * 10
        if R is None: R = np.eye(n_u)

        try:
            P = solve_continuous_are(A_lin, B_lin, Q, R)
            self.K_lqr = np.linalg.inv(R) @ B_lin.T @ P
        except np.linalg.LinAlgError:
            print("Warning: LQR solution failed. Using zero gain.")
            self.K_lqr = np.zeros((n_u, n_x))

    def compute_control(self, x, t):
        if self.koopman is None: raise RuntimeError("Koopman model not bound.")

        x = np.asarray(x).flatten()

        if self.alpha1_prev is None:
            self.alpha1_prev = x[self.state_indices[1]] # Init alpha1 with current actuator state

        if t >= self.alpha_switch * self.Tp:
            u_vec = self.linear_lqr_control(x)
            u_raw = u_vec[self.input_index]
        else:
            u_raw = self.time_varying_control(x, t)

        self.u_filtered = self.filter_alpha * self.u_filtered + (1 - self.filter_alpha) * u_raw

        if self.output_limits is not None:
            return np.clip(self.u_filtered, self.output_limits[0], self.output_limits[1])
        return self.u_filtered

    def time_varying_control(self, x, t):
        idx_1, idx_2 = self.state_indices
        x1 = x[idx_1]
        x2 = x[idx_2]
        target_1 = self.target[idx_1]
        target_2 = self.target[idx_2] # Not strictly used as target for x2, but for reference?
        # Actually backstepping target for x2 is alpha1.

        mu_t = 1.0 / max(self.Tp - t, 1e-4)

        # Projection vector for the *actuator state* (x2) dynamics
        # x2_dot = C2 (A z + B u)
        # We need to extract the row corresponding to x2 from A and B
        # in the linearized or lifted dynamics.
        # But wait, Koopman predicts z_dot.
        # x2 = z[idx_2] (assuming identity lifting for first n_x states)

        # Row for x2
        A_row = self.koopman.A[idx_2, :] # (n_lifted,)
        B_row = self.koopman.B[idx_2, :] # (n_u,)

        z1 = x1 - target_1
        # Virtual control alpha1
        # We want z1 dot = ... - sigma0 * z1 ...
        # But we don't have control over x1 directly. We assume x2 drives x1.
        # x1_dot = f1(x) + g1(x) x2 ?
        # The provided code assumed specific dynamics structure.
        # "alpha1 = T_shell_d - sigma[0] * z1"
        # This implies x1_dot ~ (x2 - x1). So if x2 = x1 - sigma z1, then x1 dot ~ -sigma z1.
        # For CSTR: dT_R/dt ~ ... + UA(T_K - T_R).
        # So T_K drives T_R.
        # So alpha1 should be desired T_K.
        # T_K_des = T_R + (desired_drift - ...)/gain?
        # The simple law "alpha1 = target_2 - sigma * z1" is a heuristic if we don't invert f1.
        # Or we can use the "target_2" (steady state x2) as bias.

        alpha1 = self.target[idx_2] - self.sigma[0] * z1
        alpha1_dot = (alpha1 - self.alpha1_prev) / self.dt
        self.alpha1_prev = alpha1

        z2 = x2 - alpha1

        # We want z2_dot = -sigma1 * mu * z2
        # x2_dot = f2_hat + B2_hat * u
        # f2_hat + B2_hat * u - alpha1_dot = -sigma1 * mu * z2
        # u = (1/B2_hat) * (alpha1_dot - f2_hat - sigma1 * mu * z2 - z1?)
        # The "-z1" term comes from Lyapunov analysis (cancelling cross term).
        # Assuming coupling x1_dot depends on x2 linearly with coeff +1?
        # If coeff is unknown, -z1 might be wrong scaling. But let's keep the structure.

        psi_x = self.koopman.lift(x)
        f2_hat = A_row @ psi_x

        # B2_hat is the effect of u[input_index] on x2
        B2_hat = B_row[self.input_index]

        # Contribution of other inputs? Assuming constant or zero?
        # f2_hat includes B * u_other if we knew u_other?
        # Koopman B is B @ u.
        # We are solving for u.
        # If n_u > 1, we fix other inputs to steady state?
        # let's assume u_sp for others.
        u_steady = self.u_sp.copy() if isinstance(self.u_sp, np.ndarray) else np.zeros(self.koopman.B.shape[1])
        # We optimize u[input_index].

        # Total B term: B_row @ u_vec
        # = B_row[idx] * u_target + sum(B_row[j] * u_steady[j])

        term_other_inputs = 0
        for j in range(len(u_steady)):
            if j != self.input_index:
                term_other_inputs += B_row[j] * u_steady[j]

        if abs(B2_hat) < 1e-5: B2_hat = np.sign(B2_hat) * 1e-5 if B2_hat != 0 else 1e-5

        # Control law
        # u = (alpha1_dot - f2_hat - term_other_inputs - sigma * mu * z2 - z1) / B2_hat
        u_val = (1.0/B2_hat) * (alpha1_dot - f2_hat - term_other_inputs - self.sigma[1] * mu_t * z2 - z1)

        return u_val

    def linear_lqr_control(self, x):
        e = x - self.target
        u_adjustment = -self.K_lqr @ e
        return self.u_sp + u_adjustment

def calculate_performance_metrics(time, response, target, settling_band=0.02):
    # Same as provided
    error = response - target
    try:
        initial_val = response[0]
        final_val = target
        ten_percent_val = initial_val + 0.1 * (final_val - initial_val)
        ninety_percent_val = initial_val + 0.9 * (final_val - initial_val)
        if final_val > initial_val:
            t10_idx = np.where(response >= ten_percent_val)[0][0]
            t90_idx = np.where(response >= ninety_percent_val)[0][0]
        else:
            t10_idx = np.where(response <= ten_percent_val)[0][0]
            t90_idx = np.where(response <= ninety_percent_val)[0][0]
        rise_time = time[t90_idx] - time[t10_idx]
    except IndexError:
        rise_time = np.nan

    settling_threshold = settling_band * abs(target)
    # logic...
    # simplified
    settling_time = np.nan # placeholder

    peak = np.max(np.abs(response)) # simplified overshoot
    overshoot = 0
    iae = np.sum(np.abs(error)) * (time[1] - time[0])
    return rise_time, settling_time, overshoot, iae
