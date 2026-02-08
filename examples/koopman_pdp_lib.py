"""
This file implements a complete example of differentiable cost learning for a
non-linear dynamical system using a Koopman operator based model predictive
controller (MPC) and a Pontryagin based parametric differentiation procedure
(PDP).  The original script has been lightly refactored and corrected to
address several numerical and theoretical issues.  In particular, the
time–varying Riccati recursion used inside the auxiliary LQR solver now
implements the standard update formula

    P_t = Q_xx - Q_ux.T @ inv(Q_uu) @ Q_ux
    W_t = Q_x  + Q_ux.T @ k_t

where ``Q_xx``, ``Q_ux``, ``Q_uu``, ``Q_x`` and ``Q_u`` are the
intermediate quantities used to build the discrete–time PMP gradients, and
``k_t = -inv(Q_uu) @ Q_u`` is the feedforward term.  The previous version
erroneously overwrote the Riccati update with a simplified form that does
not generally hold; the corrected update ensures that the auxiliary LQR
provides the true sensitivity ``∂u*/∂θ``.

The remainder of the code follows the structure laid out in the original
script: a Koopman system identification class (``Koopman_SysID``), an
optimal control system wrapper (``OCSys``), a linear quadratic regulator
solver (``LQR``) that can handle parameter–dependent terms, a coupled
oscillator environment with a slowly drifting hidden state, a simple EDMDc
fitting routine, an MPC class built on top of the learned Koopman model and
the OCSys wrapper, and finally a meta–learning loop using the PDP
derivatives to update the cost parameters.

To execute the script as a standalone experiment, run it with a Python
interpreter.  It will generate Koopman training data, fit a model,
initialize the MPC and meta–learning routines, and finally visualise
learning curves and trajectories.  All necessary dependencies are imported
locally; no external data is required.  Logging is enabled throughout
to aid debugging and reproducibility.
"""

from casadi import *  # CasADi symbols, functions and algebraic operations
import numpy as np
import logging
import time
import os
import matplotlib.pyplot as plt
# import paper_artifacts as pf  # Removed for library usage

# Configure a basic logger to capture information during execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 统一的输出目录（脚本同目录下的 artifacts）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


# ==============================================================================
# Class for Learning Koopman Models (Simplified for Proxy Generation)
# ==============================================================================
class Koopman_SysID:
    """
    Learns a discrete-time Koopman operator model (A, B) and reconstruction (C)
    from state-control trajectory data using Extended Dynamic Mode Decomposition
    with control (EDMDc).  Given a lifting function ``phi_func`` that maps
    the original state space to a higher-dimensional observable space, the
    fitted model approximates

        z_{k+1} ≈ A @ z_k + B @ u_k,
        x_k   ≈ C @ z_k,

    where ``z`` is the lifted state and ``x`` the original state.  The class
    can handle trajectories with different lengths and will skip any data
    points that contain NaNs or Infs after lifting.  A small L2 regularisation
    ``lambda_reg`` may be supplied when fitting to improve numerical
    stability.
    """

    def __init__(self, n_state, n_control, n_observables, phi_func,
                 C_matrix=None, project_name='Koopman EDMDc'):
        """
        Initialise the Koopman system identification class.
        """
        self.project_name = project_name
        self.n_state = n_state
        self.n_control = n_control
        self.n_observables = n_observables
        self.phi_func = phi_func

        # Initialise reconstruction matrix C
        if C_matrix is None:
            logger.info("No C_matrix provided, assuming C = [I | 0].")
            self.C = np.zeros((n_state, n_observables))
            if n_state <= n_observables:
                self.C[:, :n_state] = np.eye(n_state)
            else:
                logger.error(
                    f"Cannot construct default C when n_state ({n_state}) > n_observables ({n_observables})."
                )
                raise ValueError(
                    "n_state cannot exceed n_observables when C_matrix is not provided."
                )
        else:
            if not isinstance(C_matrix, np.ndarray):
                raise TypeError("C_matrix must be a numpy array.")
            if C_matrix.shape != (n_state, n_observables):
                raise ValueError(
                    f"C_matrix shape must be ({n_state}, {n_observables}), got {C_matrix.shape}."
                )
            self.C = C_matrix

        # Koopman matrices to be learned
        self.A = None
        self.B = None
        self.is_fitted_ = False

        # Validate the lifting function output dimension
        try:
            test_x = np.zeros(self.n_state)
            test_z = self.phi_func(test_x)
            if test_z.shape != (self.n_observables,):
                raise ValueError(
                    f"phi_func output dimension {test_z.shape} does not match n_observables {self.n_observables}"
                )
        except Exception as e:
            raise ValueError(f"Error testing phi_func: {e}")

    def fit(self, state_traj_list, control_traj_list, lambda_reg=1e-6):
        """
        Fit the Koopman model (A, B) using EDMDc on the provided trajectory data.
        """
        if len(state_traj_list) != len(control_traj_list):
            raise ValueError("Number of state and control trajectories must match.")

        Z_curr = []
        Z_next = []
        U_curr = []
        total_points = 0
        num_traj = len(state_traj_list)

        for i in range(num_traj):
            states = state_traj_list[i]
            controls = control_traj_list[i]

            # Skip any trajectory containing NaNs or Infs
            if np.any(np.isnan(states)) or np.any(np.isnan(controls)) or \
               np.any(np.isinf(states)) or np.any(np.isinf(controls)):
                logger.warning(f"Skipping trajectory {i} due to NaN/Inf values.")
                continue

            if states.shape[0] != controls.shape[0] + 1:
                logger.warning(
                    f"Skipping trajectory {i}: state length {states.shape[0]} != control length+1 {controls.shape[0]+1}."
                )
                continue

            if states.shape[1] != self.n_state:
                logger.warning(
                    f"Skipping trajectory {i}: state dimension {states.shape[1]} != expected {self.n_state}."
                )
                continue

            if controls.shape[1] != self.n_control:
                logger.warning(
                    f"Skipping trajectory {i}: control dimension {controls.shape[1]} != expected {self.n_control}."
                )
                continue

            horizon = controls.shape[0]
            for t in range(horizon):
                try:
                    x_k = states[t, :]
                    x_k_next = states[t + 1, :]
                    u_k = controls[t, :]

                    z_k = self.phi_func(x_k)
                    z_k_next = self.phi_func(x_k_next)

                    if (np.any(np.isnan(z_k)) or np.any(np.isnan(z_k_next)) or
                            np.any(np.isinf(z_k)) or np.any(np.isinf(z_k_next))):
                        logger.warning(
                            f"NaN/Inf detected after lifting at trajectory {i}, time {t}. Skipping point."
                        )
                        continue

                    Z_curr.append(z_k)
                    Z_next.append(z_k_next)
                    U_curr.append(u_k)
                    total_points += 1
                except Exception as e:
                    logger.error(f"Error lifting state at trajectory {i}, time {t}: {e}")
                    continue

        if total_points == 0:
            raise RuntimeError(
                "No valid lifted data points collected. Check phi_func and input data."
            )

        if total_points < self.n_observables + self.n_control:
            logger.warning(
                f"Number of data points ({total_points}) is less than the number of parameters ({self.n_observables + self.n_control}). Fit may be underdetermined."
            )

        Z_curr_mat = np.array(Z_curr).T
        Z_next_mat = np.array(Z_next).T
        U_curr_mat = np.array(U_curr).T

        Gamma = np.vstack([Z_curr_mat, U_curr_mat]).astype(np.float64)
        GGT = Gamma @ Gamma.T + lambda_reg * np.eye(self.n_observables + self.n_control)
        ZG = Z_next_mat.astype(np.float64) @ Gamma.T

        try:
            AB_T = np.linalg.solve(GGT, ZG.T)
            AB = AB_T.T
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix encountered during solve. Using pseudo-inverse.")
            AB = ZG @ np.linalg.pinv(GGT)

        self.A = AB[:, :self.n_observables]
        self.B = AB[:, self.n_observables:]
        self.is_fitted_ = True
        logger.info("Koopman model (A, B) fitted successfully.")

        try:
            residuals = Z_next_mat - (self.A @ Z_curr_mat + self.B @ U_curr_mat)
            fit_error_abs = np.linalg.norm(residuals, 'fro')
            norm_z_next = np.linalg.norm(Z_next_mat, 'fro')
            if norm_z_next > 1e-9:
                fit_error_rel = fit_error_abs / norm_z_next
                logger.info(f"Koopman fit relative Frobenius norm error: {fit_error_rel:.4e}")
            else:
                logger.info(f"Koopman fit absolute Frobenius norm error: {fit_error_abs:.4e} (Z_next norm near zero)")
        except Exception as e:
            logger.warning(f"Could not calculate fit error: {e}")
        return self

    def predict(self, ini_state_orig, control_seq):
        """
        Predict the trajectory of the original state using the fitted Koopman model.
        """
        if not self.is_fitted_:
            logger.error("Model is not fitted. Cannot predict.")
            return None

        if np.any(np.isnan(ini_state_orig)) or np.any(np.isinf(ini_state_orig)) or \
           np.any(np.isnan(control_seq)) or np.any(np.isinf(control_seq)):
            logger.error("NaN/Inf detected in input for prediction.")
            return None

        horizon = control_seq.shape[0]
        pred_state_traj_orig = np.zeros((horizon + 1, self.n_state))
        pred_state_traj_orig[0, :] = ini_state_orig.flatten()

        try:
            z_k = self.phi_func(ini_state_orig.flatten())
            if np.any(np.isnan(z_k)) or np.any(np.isinf(z_k)):
                logger.error("NaN/Inf after lifting initial state for prediction.")
                return None
        except Exception as e:
            logger.error(f"Failed to lift initial state during prediction: {e}")
            return None

        for t in range(horizon):
            try:
                u_k = control_seq[t, :]
                z_k_next = self.A @ z_k + self.B @ u_k
                if np.any(np.isnan(z_k_next)) or np.any(np.isinf(z_k_next)):
                    logger.warning(f"NaN/Inf in predicted lifted state at t={t}. Stopping prediction.")
                    pred_state_traj_orig[t + 1:, :] = np.nan
                    break

                x_k_next = self.C @ z_k_next
                pred_state_traj_orig[t + 1, :] = x_k_next
                z_k = z_k_next
            except Exception as e:
                logger.error(f"Error during prediction step t={t}: {e}")
                pred_state_traj_orig[t + 1:, :] = np.nan
                break
        return pred_state_traj_orig

    def get_model(self):
        """Return the learned Koopman model matrices."""
        if not self.is_fitted_:
            logger.warning("Model not fitted. Returning None for A, B.")
            return None, None, self.C, self.phi_func, self.n_observables
        return self.A, self.B, self.C, self.phi_func, self.n_observables


# ==============================================================================
# OCSys Class Modified for Koopman Integration
# ==============================================================================
class OCSys:
    """
    A wrapper class for building optimal control problems (OCPs) with CasADi.
    """
    def __init__(self, project_name="my optimal control system"):
        self.project_name = project_name
        self.state_orig_sym = None
        self.n_state_orig = None
        self.control_sym = None
        self.n_control = None
        self.auxvar = None
        self.n_auxvar = None
        self.state = None
        self.n_state = None
        self.control = None

        self._state_orig_lb = []
        self._state_orig_ub = []
        self.state_lb = []
        self.state_ub = []
        self.control_lb = []
        self.control_ub = []

        self.dyn = None
        self.dyn_fn = None
        self.path_cost = None
        self.path_cost_fn = None
        self.final_cost = None
        self.final_cost_fn = None

        self.using_koopman = False
        self.A_np = None
        self.B_np = None
        self.C_np = None
        self.phi_func_np = None
        self.n_lifted_state = None
        self.A_casadi = None
        self.B_casadi = None
        self.C_casadi = None

        self._pmp_diff_done = False

    def setAuxvarVariable(self, auxvar_sym):
        if auxvar_sym is None:
            logger.info("No auxiliary variable provided. Creating a dummy scalar.")
            self.auxvar = SX.sym('p_dummy')
        elif isinstance(auxvar_sym, (SX, MX)):
            self.auxvar = auxvar_sym
        else:
            raise TypeError("auxvar_sym must be None or a CasADi SX or MX symbol.")

        self.n_auxvar = self.auxvar.numel()
        logger.info(f"Auxiliary symbolic variable set (dimension {self.n_auxvar}).")
        self._pmp_diff_done = False

    def setStateVariable(self, state_orig_sym, state_lb=None, state_ub=None):
        if not isinstance(state_orig_sym, (SX, MX)):
            raise TypeError("state_orig_sym must be a CasADi symbol.")
        if self.using_koopman:
            logger.warning(
                "setStateVariable called after setKoopmanModel. Reverting to original mode."
            )
            self.using_koopman = False

        self.state_orig_sym = state_orig_sym
        self.n_state_orig = self.state_orig_sym.numel()

        if state_lb is not None and len(state_lb) == self.n_state_orig:
            self._state_orig_lb = list(state_lb)
        else:
            self._state_orig_lb = self.n_state_orig * [-1e20]

        if state_ub is not None and len(state_ub) == self.n_state_orig:
            self._state_orig_ub = list(state_ub)
        else:
            self._state_orig_ub = self.n_state_orig * [1e20]

        self.state = self.state_orig_sym
        self.n_state = self.n_state_orig
        self.state_lb = list(self._state_orig_lb)
        self.state_ub = list(self._state_orig_ub)
        logger.info(f"Original state symbolic variable set (dimension {self.n_state}).")
        self._pmp_diff_done = False

    def setControlVariable(self, control_sym, control_lb=None, control_ub=None):
        if not isinstance(control_sym, (SX, MX)):
            raise TypeError("control_sym must be a CasADi symbol.")
        self.control = control_sym
        self.n_control = self.control.numel()

        if control_lb is not None and len(control_lb) == self.n_control:
            self.control_lb = list(control_lb)
        else:
            self.control_lb = self.n_control * [-1e20]
        if control_ub is not None and len(control_ub) == self.n_control:
            self.control_ub = list(control_ub)
        else:
            self.control_ub = self.n_control * [1e20]
        logger.info(f"Control symbolic variable set (dimension {self.n_control}).")
        self._pmp_diff_done = False

    def setKoopmanModel(self, A_np, B_np, C_np, phi_func_np, n_lifted):
        if self.state_orig_sym is None or self.control is None:
            raise RuntimeError(
                "Call setStateVariable and setControlVariable before setKoopmanModel."
            )
        if not isinstance(A_np, np.ndarray) or A_np.shape != (n_lifted, n_lifted):
            raise ValueError(f"Invalid A shape {A_np.shape}, expected ({n_lifted},{n_lifted})")
        if not isinstance(B_np, np.ndarray) or B_np.shape != (n_lifted, self.n_control):
            raise ValueError(
                f"Invalid B shape {B_np.shape}, expected ({n_lifted},{self.n_control})"
            )
        if not isinstance(C_np, np.ndarray) or C_np.shape != (self.n_state_orig, n_lifted):
            raise ValueError(
                f"Invalid C shape {C_np.shape}, expected ({self.n_state_orig},{n_lifted})"
            )
        if not callable(phi_func_np):
            raise TypeError("phi_func_np must be callable")
        if n_lifted <= 0:
            raise ValueError("n_lifted must be positive.")

        self.A_np = A_np
        self.B_np = B_np
        self.C_np = C_np
        self.phi_func_np = phi_func_np
        self.n_lifted_state = n_lifted

        self.A_casadi = SX(A_np)
        self.B_casadi = SX(B_np)
        self.C_casadi = SX(C_np)

        self.state = SX.sym('z', self.n_lifted_state)
        self.n_state = self.n_lifted_state
        self.state_lb = self.n_state * [-1e20]
        self.state_ub = self.n_state * [1e20]
        self.using_koopman = True
        logger.info(
            f"Koopman mode activated. Lifted state dimension {self.n_state}. Original state bounds ignored."
        )
        self._pmp_diff_done = False

    def setDyn(self, ode_expr=None):
        if self.state is None or self.control is None:
            raise RuntimeError("Call setStateVariable and setControlVariable first.")
        if self.auxvar is None:
            self.setAuxvarVariable(None)

        if self.using_koopman:
            if self.A_casadi is None or self.B_casadi is None:
                raise RuntimeError("Koopman matrices not set via setKoopmanModel.")
            self.dyn = mtimes(self.A_casadi, self.state) + mtimes(self.B_casadi, self.control)
            self.dyn_fn = Function(
                'koopman_dynamics',
                [self.state, self.control, self.auxvar],
                [self.dyn],
                ['z', 'u', 'p'],
                ['z_next']
            )
            logger.info("Koopman dynamics set: z_next = A z + B u.")
        else:
            if ode_expr is None:
                raise ValueError("ode_expr must be provided if not using Koopman mode.")
            if not isinstance(ode_expr, (SX, MX)):
                raise TypeError("ode_expr must be a CasADi SX or MX expression.")
            try:
                Function('dyn_test', [self.state, self.control, self.auxvar], [ode_expr])
            except Exception as e:
                raise ValueError(
                    f"Error in provided ode_expr. Ensure it uses the correct symbolic variables. CasADi error: {e}"
                )
            self.dyn = ode_expr
            self.dyn_fn = Function(
                'dynamics',
                [self.state, self.control, self.auxvar],
                [self.dyn],
                ['x', 'u', 'p'],
                ['x_next']
            )
            logger.info("Original dynamics set.")
        self._pmp_diff_done = False

    def _create_cost_func(self, cost_lambda, cost_type="path"):
        if self.state is None or self.control is None or self.auxvar is None:
            raise RuntimeError("State, control and auxvar must be set before defining costs.")
        if self.state_orig_sym is None:
            raise RuntimeError("Original state symbol required for cost definition.")

        if cost_type == "path":
            lambda_args_sym = [self.state_orig_sym, self.control, self.auxvar]
            input_vars_fn = [self.state, self.control, self.auxvar]
            func_name_prefix = "path_cost"
            input_names_fn = ['state', 'u', 'p']
            output_name_fn = 'path_cost_val'
        elif cost_type == "final":
            lambda_args_sym = [self.state_orig_sym, self.auxvar]
            input_vars_fn = [self.state, self.auxvar]
            func_name_prefix = "final_cost"
            input_names_fn = ['state', 'p']
            output_name_fn = 'final_cost_val'
        else:
            raise ValueError(f"Invalid cost_type: {cost_type}")

        try:
            cost_expr_orig_sym = cost_lambda(*lambda_args_sym)
            if not isinstance(cost_expr_orig_sym, (SX, MX)):
                raise TypeError(f"Cost lambda for {cost_type} did not return a CasADi expression.")
            if cost_expr_orig_sym.numel() != 1:
                raise ValueError(f"{cost_type} cost lambda must return a scalar expression.")
        except Exception as e:
            raise ValueError(f"Error evaluating cost lambda function: {e}")

        if self.using_koopman:
            if self.C_casadi is None:
                raise RuntimeError("Koopman C matrix not set.")
            x_reconstructed = mtimes(self.C_casadi, self.state)
            cost_expr = substitute(cost_expr_orig_sym, self.state_orig_sym, x_reconstructed)
            func_name = f"koopman_{func_name_prefix}_transformed"
        else:
            cost_expr = cost_expr_orig_sym
            func_name = f"original_{func_name_prefix}"

        cost_fn = Function(
            func_name,
            input_vars_fn,
            [cost_expr],
            input_names_fn,
            [output_name_fn]
        )
        return cost_expr, cost_fn

    def setPathCost(self, path_cost_lambda):
        if self.state_orig_sym is None or self.control is None or self.auxvar is None:
            raise RuntimeError("Set state, control and auxvar before defining path cost.")
        self.path_cost, self.path_cost_fn = self._create_cost_func(path_cost_lambda, "path")
        logger.info(f"Path cost set {'(transformed for Koopman)' if self.using_koopman else ''}.")
        self._pmp_diff_done = False

    def setFinalCost(self, final_cost_lambda):
        if self.state_orig_sym is None or self.auxvar is None:
            raise RuntimeError("Set state and auxvar before defining final cost.")
        self.final_cost, self.final_cost_fn = self._create_cost_func(final_cost_lambda, "final")
        logger.info(f"Final cost set {'(transformed for Koopman)' if self.using_koopman else ''}.")
        self._pmp_diff_done = False

    def ocSolver(self, ini_state_orig, horizon, auxvar_value=None, print_level=0, costate_option=0):
        """
        Solve the optimal control problem using a multiple shooting formulation.
        """
        if self.state is None or self.control is None or self.dyn_fn is None or \
           self.path_cost_fn is None or self.final_cost_fn is None:
            raise RuntimeError("Dynamics and costs must be set before calling ocSolver.")
        if horizon <= 0:
            raise ValueError("Horizon must be positive.")
        if isinstance(ini_state_orig, list):
            ini_state_orig_np = np.array(ini_state_orig, dtype=float).flatten()
        elif isinstance(ini_state_orig, np.ndarray):
            ini_state_orig_np = ini_state_orig.flatten()
        else:
            raise TypeError("ini_state_orig must be a list or numpy array.")
        if ini_state_orig_np.shape != (self.n_state_orig,):
            raise ValueError(
                f"ini_state_orig dimension mismatch: got {ini_state_orig_np.shape}, expected {(self.n_state_orig,)}"
            )

        if auxvar_value is None:
            if self.n_auxvar > 0:
                raise ValueError("auxvar_value must be provided when n_auxvar > 0.")
            auxvar_value_np = np.array([])
        elif isinstance(auxvar_value, (int, float)):
            auxvar_value_np = np.array([float(auxvar_value)])
        elif isinstance(auxvar_value, (list, np.ndarray)):
            auxvar_value_np = np.array(auxvar_value, dtype=float).flatten()
        else:
            raise TypeError("auxvar_value must be float, list or numpy array.")
        if self.n_auxvar != auxvar_value_np.size:
            if self.n_auxvar == 0:
                auxvar_value_np = np.array([])
            else:
                raise ValueError(
                    f"auxvar_value dimension mismatch: got {auxvar_value_np.size}, expected {self.n_auxvar}."
                )

        if self.using_koopman:
            if self.phi_func_np is None:
                raise RuntimeError("Lifting function not set for Koopman mode.")
            ini_state_lifted_np = self.phi_func_np(ini_state_orig_np)
            if ini_state_lifted_np.shape != (self.n_lifted_state,):
                raise RuntimeError(
                    f"phi_func_np output dimension mismatch: got {ini_state_lifted_np.shape}, expected {(self.n_lifted_state,)}"
                )
            ini_state_for_nlp = ini_state_lifted_np.tolist()
            current_n_state_nlp = self.n_lifted_state
            state_lb_nlp = self.state_lb
            state_ub_nlp = self.state_ub
        else:
            ini_state_for_nlp = ini_state_orig_np.tolist()
            current_n_state_nlp = self.n_state_orig
            state_lb_nlp = self.state_lb
            state_ub_nlp = self.state_ub

        w = []
        w0 = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []
        J = 0

        Xk = MX.sym('X0', current_n_state_nlp)
        w.append(Xk)
        lbw += ini_state_for_nlp
        ubw += ini_state_for_nlp
        w0 += ini_state_for_nlp

        for k in range(horizon):
            Uk = MX.sym(f'U_{k}', self.n_control)
            w.append(Uk)
            lbw += self.control_lb
            ubw += self.control_ub
            u_guess = [0.5 * (lb + ub) if np.isfinite(lb) and np.isfinite(ub) else 0.0
                       for lb, ub in zip(self.control_lb, self.control_ub)]
            w0 += u_guess

            Xk_next_dyn = self.dyn_fn(Xk, Uk, auxvar_value_np if self.n_auxvar > 0 else SX())
            path_cost_k = self.path_cost_fn(Xk, Uk, auxvar_value_np if self.n_auxvar > 0 else SX())
            J += path_cost_k

            Xk_next_var = MX.sym(f'X_{k + 1}', current_n_state_nlp)
            w.append(Xk_next_var)
            lbw += state_lb_nlp
            ubw += state_ub_nlp
            x_guess = [0.5 * (lb + ub) if np.isfinite(lb) and np.isfinite(ub) else ini_state_for_nlp[idx] if idx < len(ini_state_for_nlp) else 0.0
                       for idx, (lb, ub) in enumerate(zip(state_lb_nlp, state_ub_nlp))]
            w0 += x_guess

            g.append(Xk_next_dyn - Xk_next_var)
            lbg += current_n_state_nlp * [0]
            ubg += current_n_state_nlp * [0]

            Xk = Xk_next_var

        J += self.final_cost_fn(Xk, auxvar_value_np if self.n_auxvar > 0 else SX())

        nlp_prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        ipopt_opts = {
            'ipopt.print_level': print_level,
            'ipopt.sb': 'yes',
            'print_time': bool(print_level > 0)
        }
        solver = nlpsol('solver', 'ipopt', nlp_prob, ipopt_opts)

        try:
            expected_w_len = current_n_state_nlp * (horizon + 1) + self.n_control * horizon
            if len(w0) != expected_w_len:
                logger.error(
                    f"Initial guess length {len(w0)} != expected {expected_w_len}. Adjusting to zeros."
                )
                w0 = [0.0] * expected_w_len
            sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        except Exception as e:
            logger.error(f"Error during NLP solve: {e}")
            return {
                "control_traj_opt": None,
                "costate_traj_opt": None,
                'auxvar_value': auxvar_value_np,
                "time": None,
                "horizon": horizon,
                "cost": float('inf'),
                "success": False,
                "solver_stats": {}
            }

        w_opt_flat = sol['x'].full().flatten()
        cost_opt = float(sol['f'])
        lam_g_flat = sol['lam_g'].full().flatten()
        solver_stats = solver.stats()
        success = solver_stats.get('success', False)
        if not success:
            logger.warning(f"IPOPT solver did not converge: {solver_stats.get('return_status', 'N/A')}")

        state_traj_raw = np.zeros((horizon + 1, current_n_state_nlp))
        control_traj_opt = np.zeros((horizon, self.n_control))
        try:
            state_traj_raw[0, :] = w_opt_flat[:current_n_state_nlp]
            offset = current_n_state_nlp
            for k in range(horizon):
                control_traj_opt[k, :] = w_opt_flat[offset: offset + self.n_control]
                offset += self.n_control
                state_traj_raw[k + 1, :] = w_opt_flat[offset: offset + current_n_state_nlp]
                offset += current_n_state_nlp
            if offset != len(w_opt_flat):
                logger.warning(
                    f"Offset {offset} doesn't match solution length {len(w_opt_flat)} after extraction."
                )
        except IndexError as e:
            logger.error(f"Error extracting solution: {e}")
            success = False

        time_vec = np.arange(horizon + 1)
        opt_sol = {
            "control_traj_opt": control_traj_opt if success else None,
            "costate_traj_opt": None,
            'auxvar_value': auxvar_value_np,
            "time": time_vec if success else None,
            "horizon": horizon,
            "cost": cost_opt if success else float('inf'),
            "success": success,
            "solver_stats": solver_stats
        }

        if success:
            if self.using_koopman:
                opt_sol["lifted_state_traj_opt"] = state_traj_raw
                try:
                    reconstructed_state_traj = (self.C_np @ state_traj_raw.T).T
                    opt_sol["state_traj_opt"] = reconstructed_state_traj
                except Exception as e:
                    logger.error(f"Failed to reconstruct original state trajectory: {e}")
                    opt_sol["state_traj_opt"] = None
                    opt_sol["success"] = False
            else:
                opt_sol["state_traj_opt"] = state_traj_raw
                opt_sol["lifted_state_traj_opt"] = None
        else:
            opt_sol["state_traj_opt"] = None
            opt_sol["lifted_state_traj_opt"] = None

        if success:
            costate_traj_nlp = np.reshape(lam_g_flat, (horizon, current_n_state_nlp))
        else:
            costate_traj_nlp = None

        if costate_option == 0:
            opt_sol["costate_traj_opt"] = costate_traj_nlp
        elif costate_option == 1 and success:
            if not self._pmp_diff_done:
                self.diffPMP()
            try:
                dcx_wrt_state_fn = Function(
                    'dcx_wrt_state',
                    [self.state, self.control, self.auxvar],
                    [jacobian(self.path_cost, self.state)],
                )
                dhx_fn_internal = self.dhx_fn
                dfx_fn_internal = self.dfx_fn

                costate_traj_pmp = np.zeros((horizon + 1, current_n_state_nlp))
                costate_traj_pmp[horizon, :] = dhx_fn_internal(state_traj_raw[horizon, :],
                                                               auxvar_value_np if self.n_auxvar > 0 else SX()).full().flatten()

                for k in range(horizon - 1, -1, -1):
                    state_k = state_traj_raw[k, :]
                    control_k = control_traj_opt[k, :]
                    costate_k_plus_1 = costate_traj_pmp[k + 1, :]

                    dcx_k = dcx_wrt_state_fn(state_k, control_k,
                                             auxvar_value_np if self.n_auxvar > 0 else SX()).full().T
                    dfx_k = dfx_fn_internal(state_k, control_k,
                                             auxvar_value_np if self.n_auxvar > 0 else SX()).full()
                    costate_k = dcx_k.T + costate_k_plus_1 @ dfx_k
                    costate_traj_pmp[k, :] = costate_k.flatten()
                opt_sol["costate_traj_opt"] = costate_traj_pmp[1:, :]
            except Exception as e:
                logger.error(f"Error computing costates via PMP: {e}")
                opt_sol["costate_traj_opt"] = costate_traj_nlp
        elif costate_option == 1 and not success:
            logger.warning("Cannot compute costates via PMP because NLP failed.")
            opt_sol["costate_traj_opt"] = costate_traj_nlp

        if opt_sol["costate_traj_opt"] is None and success:
            logger.warning("Costate calculation resulted in None despite solver success.")
            opt_sol["success"] = False

        return opt_sol

    def diffPMP(self):
        """
        Compute derivatives required for the PMP backward recursion and the
        auxiliary system.
        """
        if self.state is None or self.control is None or self.auxvar is None or \
           self.dyn is None or self.path_cost is None or self.final_cost is None:
            raise RuntimeError(
                "Cannot differentiate PMP. Ensure state, control, auxvar, dynamics and costs are set."
            )

        logger.debug(f"Computing PMP derivatives {'(Koopman)' if self.using_koopman else ''}...")

        try:
            self.costate = SX.sym('lambda', self.n_state)
            self.path_Hamil = self.path_cost + dot(self.costate, self.dyn)
            self.final_Hamil = self.final_cost

            # Dynamics derivatives
            self.dfx = jacobian(self.dyn, self.state)
            self.dfx_fn = Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
            self.dfu = jacobian(self.dyn, self.control)
            self.dfu_fn = Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
            self.dfe = jacobian(self.dyn, self.auxvar) if self.n_auxvar > 0 else SX.zeros(self.n_state, 0)
            self.dfe_fn = Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])

            # Hamiltonian derivatives
            self.dHx = jacobian(self.path_Hamil, self.state).T
            self.dHx_fn = Function('dHx', [self.state, self.control, self.costate, self.auxvar], [self.dHx])
            self.dHu = jacobian(self.path_Hamil, self.control).T
            self.dHu_fn = Function('dHu', [self.state, self.control, self.costate, self.auxvar], [self.dHu])

            # Second derivatives
            self.ddHxx = jacobian(self.dHx, self.state)
            self.ddHxx_fn = Function('ddHxx', [self.state, self.control, self.costate, self.auxvar], [self.ddHxx])
            self.ddHxu = jacobian(self.dHx, self.control)
            self.ddHxu_fn = Function('ddHxu', [self.state, self.control, self.costate, self.auxvar], [self.ddHxu])
            self.ddHxe = jacobian(self.dHx, self.auxvar) if self.n_auxvar > 0 else SX.zeros(self.n_state, self.n_auxvar)
            self.ddHxe_fn = Function('ddHxe', [self.state, self.control, self.costate, self.auxvar], [self.ddHxe])
            self.ddHux = jacobian(self.dHu, self.state)
            self.ddHux_fn = Function('ddHux', [self.state, self.control, self.costate, self.auxvar], [self.ddHux])
            self.ddHuu = jacobian(self.dHu, self.control)
            self.ddHuu_fn = Function('ddHuu', [self.state, self.control, self.costate, self.auxvar], [self.ddHuu])
            self.ddHue = jacobian(self.dHu, self.auxvar) if self.n_auxvar > 0 else SX.zeros(self.n_control, self.n_auxvar)
            self.ddHue_fn = Function('ddHue', [self.state, self.control, self.costate, self.auxvar], [self.ddHue])

            # Final cost derivatives
            self.dhx = jacobian(self.final_Hamil, self.state).T
            self.dhx_fn = Function('dhx', [self.state, self.auxvar], [self.dhx])
            self.ddhxx = jacobian(self.dhx, self.state)
            self.ddhxx_fn = Function('ddhxx', [self.state, self.auxvar], [self.ddhxx])
            self.ddhxe = jacobian(self.dhx, self.auxvar) if self.n_auxvar > 0 else SX.zeros(self.n_state, self.n_auxvar)
            self.ddhxe_fn = Function('ddhxe', [self.state, self.auxvar], [self.ddhxe])

            self._pmp_diff_done = True
            logger.debug("PMP derivatives computed.")
        except Exception as e:
            logger.error(f"Error during PMP differentiation: {e}")
            self._pmp_diff_done = False
            raise

    def getAuxSys(self, state_traj_opt, control_traj_opt, costate_traj_opt, auxvar_value=None):
        """
        Assemble the time-varying matrices required for the auxiliary LQR
        system.
        """
        if not self._pmp_diff_done:
            try:
                self.diffPMP()
            except Exception as e:
                logger.error(f"Failed to compute PMP derivatives for getAuxSys: {e}")
                return None

        horizon = control_traj_opt.shape[0]
        if state_traj_opt.shape != (horizon + 1, self.n_state):
            logger.error(
                f"State trajectory shape mismatch: got {state_traj_opt.shape}, expected {(horizon+1, self.n_state)}"
            )
            return None
        if costate_traj_opt is None:
            logger.error("Costate trajectory is None in getAuxSys.")
            return None
        if costate_traj_opt.shape != (horizon, self.n_state):
            logger.error(
                f"Costate trajectory shape mismatch: got {costate_traj_opt.shape}, expected {(horizon, self.n_state)}"
            )
            return None
        if control_traj_opt.shape[1] != self.n_control:
            logger.error("Control trajectory dimension mismatch in getAuxSys.")
            return None
        if np.any(np.isnan(state_traj_opt)) or np.any(np.isnan(control_traj_opt)) or np.any(np.isnan(costate_traj_opt)):
            logger.error("NaN detected in input trajectories for getAuxSys.")
            return None

        if auxvar_value is None:
            auxvar_value_np = np.array([]) if self.n_auxvar == 0 else None
            if self.n_auxvar > 0:
                raise ValueError("auxvar_value must be provided if n_auxvar > 0")
        elif isinstance(auxvar_value, (int, float)):
            auxvar_value_np = np.array([float(auxvar_value)])
        elif isinstance(auxvar_value, (list, np.ndarray)):
            auxvar_value_np = np.array(auxvar_value, dtype=float).flatten()
        else:
            raise TypeError("auxvar_value type invalid.")
        if self.n_auxvar != auxvar_value_np.size:
            if self.n_auxvar == 0:
                auxvar_value_np = np.array([])
            else:
                raise ValueError(
                    f"auxvar_value dimension mismatch: got {auxvar_value_np.size}, expected {self.n_auxvar}."
                )

        dynF, dynG, dynE = [], [], []
        Hxx, Hxu, Hxe, Hux, Huu, Hue = [], [], [], [], [], []

        logger.debug("Computing auxiliary system matrices along the optimal trajectory.")
        try:
            empty_aux_arg = SX()
            for t in range(horizon):
                state_t = state_traj_opt[t, :]
                control_t = control_traj_opt[t, :]
                costate_t_plus_1 = costate_traj_opt[t, :]

                aux_arg = auxvar_value_np if self.n_auxvar > 0 else empty_aux_arg

                dynF.append(self.dfx_fn(state_t, control_t, aux_arg).full())
                dynG.append(self.dfu_fn(state_t, control_t, aux_arg).full())
                dynE.append(self.dfe_fn(state_t, control_t, aux_arg).full())

                Hxx.append(self.ddHxx_fn(state_t, control_t, costate_t_plus_1, aux_arg).full())
                Hxu.append(self.ddHxu_fn(state_t, control_t, costate_t_plus_1, aux_arg).full())
                Hxe.append(self.ddHxe_fn(state_t, control_t, costate_t_plus_1, aux_arg).full())
                Hux.append(self.ddHux_fn(state_t, control_t, costate_t_plus_1, aux_arg).full())
                Huu.append(self.ddHuu_fn(state_t, control_t, costate_t_plus_1, aux_arg).full())
                Hue.append(self.ddHue_fn(state_t, control_t, costate_t_plus_1, aux_arg).full())

            final_state = state_traj_opt[-1, :]
            aux_arg_final = auxvar_value_np if self.n_auxvar > 0 else empty_aux_arg
            hxx = [self.ddhxx_fn(final_state, aux_arg_final).full()]
            hxe = [self.ddhxe_fn(final_state, aux_arg_final).full()]

            if self.n_auxvar == 0:
                dynE = [np.zeros((self.n_state, 0)) for _ in range(horizon)]
                Hxe = [np.zeros((self.n_state, 0)) for _ in range(horizon)]
                Hue = [np.zeros((self.n_control, 0)) for _ in range(horizon)]
                hxe = [np.zeros((self.n_state, 0))]

        except Exception as e:
            logger.error(f"Error evaluating derivatives in getAuxSys: {e}")
            raise

        return {
            "dynF": dynF, "dynG": dynG, "dynE": dynE,
            "Hxx": Hxx, "Hxu": Hxu, "Hxe": Hxe,
            "Hux": Hux, "Huu": Huu, "Hue": Hue,
            "hxx": hxx, "hxe": hxe
        }


# ==============================================================================
# LQR Class (Corrected and Enhanced Validation)
# ==============================================================================
class LQR:
    """
    Solve a time–varying linear quadratic regulator problem.
    """
    def __init__(self, project_name="LQR system"):
        self.project_name = project_name
        self.n_state = None
        self.n_control = None
        self.n_batch = None
        self.horizon = None
        self.dynF = None
        self.dynG = None
        self.dynE = None
        self.Hxx = None
        self.Huu = None
        self.Hxu = None
        self.Hux = None
        self.Hxe = None
        self.Hue = None
        self.hxx = None
        self.hxe = None

    def _validate_matrix_list(self, matrix_list, name, shape_tpl, expected_len,
                              allow_none=False, is_final_cost=False, current_n_batch=None):
        if matrix_list is None:
            if allow_none:
                return None, current_n_batch
            else:
                raise TypeError(f"{name} cannot be None.")
        if is_final_cost and isinstance(matrix_list, np.ndarray):
            matrix_list = [matrix_list]
        elif not isinstance(matrix_list, list):
            raise TypeError(f"{name} must be a list of numpy arrays.")
        actual_len = len(matrix_list)
        if is_final_cost:
            if actual_len != 1:
                raise ValueError(f"{name} list must have length 1 for final cost, got {actual_len}.")
        else:
            if actual_len != expected_len:
                raise ValueError(
                    f"{name} length {actual_len} != expected horizon {expected_len}."
                )
        first_matrix = matrix_list[0]
        if not isinstance(first_matrix, np.ndarray):
            raise TypeError(f"Elements of {name} must be numpy arrays.")
        if first_matrix.ndim != 2:
            is_param_term = name in ['dynE', 'Hxe', 'Hue', 'hxe']
            if is_param_term and current_n_batch == 0 and first_matrix.shape == (shape_tpl[0],):
                matrix_list = [m.reshape(shape_tpl[0], 0) for m in matrix_list]
                first_matrix = matrix_list[0]
            else:
                raise ValueError(
                    f"{name} must be a 2D array, but element 0 has shape {first_matrix.shape}."
                )
        actual_rows, actual_cols = first_matrix.shape
        expected_rows, expected_cols = shape_tpl
        if expected_rows != -1 and actual_rows != expected_rows:
            raise ValueError(
                f"{name}[0] row dimension mismatch: expected {expected_rows}, got {actual_rows}."
            )
        if expected_cols != -1 and actual_cols != expected_cols:
            is_param_term = name in ['dynE', 'Hxe', 'Hue', 'hxe']
            if is_param_term and current_n_batch is not None and expected_cols == -1:
                expected_cols = current_n_batch
                if actual_cols != expected_cols:
                    raise ValueError(
                        f"{name}[0] column dimension mismatch: expected {expected_cols}, got {actual_cols}."
                    )
            elif actual_cols != expected_cols:
                raise ValueError(
                    f"{name}[0] column dimension mismatch: expected {expected_cols}, got {actual_cols}."
                )
        n_batch_updated = current_n_batch
        if name in ['dynE', 'Hxe', 'Hue', 'hxe']:
            _n_batch_term = actual_cols
            if n_batch_updated is None:
                n_batch_updated = _n_batch_term
                logger.debug(f"Inferred n_batch = {_n_batch_term} from {name}.")
            elif n_batch_updated != _n_batch_term:
                raise ValueError(
                    f"n_batch mismatch in {name} ({_n_batch_term}) vs previous ({n_batch_updated})."
                )
        if len(matrix_list) > 1:
            expected_shape = first_matrix.shape
            for i, m in enumerate(matrix_list[1:], start=1):
                if m.shape != expected_shape:
                    raise ValueError(
                        f"{name} matrices have inconsistent shapes: element 0 has {expected_shape}, element {i} has {m.shape}."
                    )
        return matrix_list, n_batch_updated

    def setDyn(self, dynF, dynG, dynE=None):
        if not isinstance(dynF, list) or not isinstance(dynG, list):
            raise TypeError("dynF and dynG must be lists.")
        if len(dynF) == 0 or len(dynG) == 0:
            raise ValueError("dynF and dynG cannot be empty.")
        if len(dynF) != len(dynG):
            raise ValueError("dynF and dynG must have the same length.")
        self.horizon = len(dynF)
        self.n_batch = None
        self.dynF, _ = self._validate_matrix_list(dynF, 'dynF', (-1, -1), self.horizon, False)
        self.n_state = self.dynF[0].shape[0]
        if self.dynF[0].shape[1] != self.n_state:
            raise ValueError("dynF matrices must be square.")
        self.dynG, _ = self._validate_matrix_list(dynG, 'dynG', (self.n_state, -1), self.horizon, False)
        self.n_control = self.dynG[0].shape[1]
        self.dynE, self.n_batch = self._validate_matrix_list(dynE, 'dynE', (self.n_state, -1), self.horizon, True, False, self.n_batch)
        logger.debug(
            f"LQR dynamics set: n_state={self.n_state}, n_control={self.n_control}, n_batch={self.n_batch}, horizon={self.horizon}."
        )

    def setPathCost(self, Hxx, Huu, Hxu=None, Hux=None, Hxe=None, Hue=None):
        if self.n_state is None or self.n_control is None or self.horizon is None:
            raise RuntimeError("Call setDyn first.")
        self.Hxx, self.n_batch = self._validate_matrix_list(Hxx, 'Hxx', (self.n_state, self.n_state), self.horizon, False, False, self.n_batch)
        self.Huu, self.n_batch = self._validate_matrix_list(Huu, 'Huu', (self.n_control, self.n_control), self.horizon, False, False, self.n_batch)
        self.Hxu, self.n_batch = self._validate_matrix_list(Hxu, 'Hxu', (self.n_state, self.n_control), self.horizon, True, False, self.n_batch)
        self.Hux, self.n_batch = self._validate_matrix_list(Hux, 'Hux', (self.n_control, self.n_state), self.horizon, True, False, self.n_batch)
        self.Hxe, self.n_batch = self._validate_matrix_list(Hxe, 'Hxe', (self.n_state, -1), self.horizon, True, False, self.n_batch)
        self.Hue, self.n_batch = self._validate_matrix_list(Hue, 'Hue', (self.n_control, -1), self.horizon, True, False, self.n_batch)
        if self.n_batch is None:
            logger.debug("n_batch not set by dynamics or costs. Defaulting to 0.")
            self.n_batch = 0
        if self.dynE is None:
            self.dynE = [np.zeros((self.n_state, self.n_batch)) for _ in range(self.horizon)]
        elif self.dynE[0].shape[1] != self.n_batch:
            raise ValueError(
                f"n_batch mismatch: dynE columns {self.dynE[0].shape[1]} vs inferred {self.n_batch}."
            )
        logger.debug(f"LQR stage costs set. n_batch confirmed as {self.n_batch}.")

    def setFinalCost(self, hxx, hxe=None):
        if self.n_state is None:
            raise RuntimeError("Call setDyn first.")
        if self.horizon is None:
            raise RuntimeError("Horizon not set (call setDyn).")
        self.hxx, self.n_batch = self._validate_matrix_list(hxx, 'hxx', (self.n_state, self.n_state), 1, False, True, self.n_batch)
        self.hxe, self.n_batch = self._validate_matrix_list(hxe, 'hxe', (self.n_state, -1), 1, True, True, self.n_batch)
        if self.n_batch is None:
            logger.warning("n_batch could not be inferred, defaulting to 0.")
            self.n_batch = 0
        if self.dynE is None:
            self.dynE = [np.zeros((self.n_state, self.n_batch)) for _ in range(self.horizon)]
        elif self.dynE[0].shape[1] != self.n_batch:
            raise ValueError(
                f"n_batch conflict: dynE columns {self.dynE[0].shape[1]} vs final cost n_batch {self.n_batch}."
            )
        if self.hxe is None:
            self.hxe = [np.zeros((self.n_state, self.n_batch))]
        elif self.hxe[0].shape[1] != self.n_batch:
            raise ValueError(
                f"hxe columns mismatch: got {self.hxe[0].shape[1]}, expected {self.n_batch}."
            )
        logger.debug(f"LQR terminal cost set. n_batch finalised as {self.n_batch}.")

    def lqrSolver(self, ini_state, horizon):
        """
        Solve the LQR problem by backward Riccati + forward simulation.
        """
        if self.dynF is None or self.Hxx is None or self.hxx is None:
            raise RuntimeError("Dynamics and costs must be set before solving.")
        if self.horizon is None or self.horizon != horizon:
            raise ValueError(
                f"Provided horizon {horizon} does not match the configured horizon {self.horizon}."
            )
        if self.n_batch is None:
            raise RuntimeError("n_batch could not be inferred. Check inputs.")

        if isinstance(ini_state, list):
            ini_x_np = np.array(ini_state, dtype=float)
        elif isinstance(ini_state, np.ndarray):
            ini_x_np = ini_state.astype(float)
        else:
            raise TypeError("ini_state must be list or numpy array.")
        if ini_x_np.ndim == 1:
            if ini_x_np.shape[0] != self.n_state:
                raise ValueError(
                    f"ini_state dimension mismatch: got {ini_x_np.shape[0]}, expected {self.n_state}."
                )
            _n_batch_ini = 1
            ini_x_batch = ini_x_np.reshape(self.n_state, 1)
        elif ini_x_np.ndim == 2:
            if ini_x_np.shape[0] != self.n_state:
                raise ValueError(
                    f"ini_state row dimension mismatch: got {ini_x_np.shape[0]}, expected {self.n_state}."
                )
            _n_batch_ini = ini_x_np.shape[1]
            ini_x_batch = ini_x_np
        else:
            raise ValueError("ini_state must be 1D or 2D.")
        if self.n_batch > 0 and self.n_batch != _n_batch_ini:
            raise ValueError(
                f"n_batch mismatch: parameter dim {self.n_batch} vs initial state batches {_n_batch_ini}."
            )
        elif self.n_batch == 0:
            if not all(m.shape[1] == 0 for m in self.dynE):
                raise RuntimeError("Internal error: dynE not shaped correctly for n_batch=0")
            if self.Hxe is not None and not all(m.shape[1] == 0 for m in self.Hxe):
                raise RuntimeError("Internal error: Hxe not shaped correctly for n_batch=0")
            if self.Hue is not None and not all(m.shape[1] == 0 for m in self.Hue):
                raise RuntimeError("Internal error: Hue not shaped correctly for n_batch=0")
            if self.hxe is not None and self.hxe[0].shape[1] != 0:
                raise RuntimeError("Internal error: hxe not shaped correctly for n_batch=0")

        F, G, E = self.dynF, self.dynG, self.dynE
        Hxx, Huu = self.Hxx, self.Huu
        Hxu = self.Hxu if self.Hxu is not None else [np.zeros((self.n_state, self.n_control)) for _ in range(horizon)]
        Hux = self.Hux if self.Hux is not None else [np.zeros((self.n_control, self.n_state)) for _ in range(horizon)]
        Hxe = self.Hxe if self.Hxe is not None else [np.zeros((self.n_state, self.n_batch)) for _ in range(horizon)]
        Hue = self.Hue if self.Hue is not None else [np.zeros((self.n_control, self.n_batch)) for _ in range(horizon)]
        hxx_final = self.hxx[0]
        hxe_final = self.hxe[0]

        PP = [None] * (horizon + 1)
        WW = [None] * (horizon + 1)
        PP[horizon] = hxx_final
        WW[horizon] = hxe_final
        K_feedback = [None] * horizon
        k_feedforward = [None] * horizon

        for t in range(horizon - 1, -1, -1):
            Ft, Gt, Et = F[t], G[t], E[t]
            Qt, Rut = Hxx[t], Huu[t]
            P_next, W_next = PP[t + 1], WW[t + 1]

            Q_uu = Rut + Gt.T @ P_next @ Gt
            Q_ux = Hux[t] + Gt.T @ P_next @ Ft
            Q_u = Hue[t] + Gt.T @ W_next + Gt.T @ P_next @ Et
            Q_xx = Qt + Ft.T @ P_next @ Ft
            Q_x = Hxe[t] + Ft.T @ W_next + Ft.T @ P_next @ Et

            try:
                inv_Quu = np.linalg.inv(Q_uu)
            except np.linalg.LinAlgError:
                logger.warning(f"Q_uu singular at t={t}. Using pseudo-inverse.")
                inv_Quu = np.linalg.pinv(Q_uu)

            K_t = -inv_Quu @ Q_ux
            k_t = -inv_Quu @ Q_u

            P_t = Q_xx - Q_ux.T @ inv_Quu @ Q_ux
            W_t = Q_x + Q_ux.T @ k_t

            PP[t] = P_t
            WW[t] = W_t
            K_feedback[t] = K_t
            k_feedforward[t] = k_t

        state_traj_opt = [None] * (horizon + 1)
        control_traj_opt = [None] * horizon
        costate_traj_opt = [None] * (horizon + 1)
        state_traj_opt[0] = ini_x_batch
        costate_traj_opt[0] = PP[0] @ state_traj_opt[0] + WW[0]
        for t in range(horizon):
            Ft, Gt, Et = F[t], G[t], E[t]
            x_t = state_traj_opt[t]
            K_t = K_feedback[t]
            k_t = k_feedforward[t]
            if self.n_batch == 0 and _n_batch_ini > 0:
                u_t = K_t @ x_t
            else:
                u_t = K_t @ x_t + k_t
            if self.n_batch == 0 and _n_batch_ini > 0:
                x_next = Ft @ x_t + Gt @ u_t
            else:
                x_next = Ft @ x_t + Gt @ u_t + Et
            P_next = PP[t + 1]
            W_next = WW[t + 1]
            if self.n_batch == 0 and _n_batch_ini > 0:
                lambda_next = P_next @ x_next
            else:
                lambda_next = P_next @ x_next + W_next
            state_traj_opt[t + 1] = x_next
            control_traj_opt[t] = u_t
            costate_traj_opt[t + 1] = lambda_next
        time_vec = np.arange(horizon + 1)
        return {
            'state_traj_opt': state_traj_opt,
            'control_traj_opt': control_traj_opt,
            'costate_traj_opt': costate_traj_opt[1:],
            'time': time_vec
        }


# ==============================================================================
# Koopman Model Identification (Generic)
# ==============================================================================
def identify_koopman_model(system, state_traj_list_obs, control_traj_list, phi_func=None):
    logger.info("Identifying Koopman model...")

    if phi_func is None:
        def default_phi_func(x_obs_np):
            # Default quadratic lifting
            poly = [x_obs_np]
            n = len(x_obs_np)
            # Quadratic terms
            for i in range(n):
                for j in range(i, n):
                    poly.append(x_obs_np[i] * x_obs_np[j])
            return np.hstack(poly)
        phi_func = default_phi_func

    # Infer n_observables
    try:
        test_x = np.zeros(system.n_state_obs)
        n_observables = phi_func(test_x).shape[0]
    except Exception as e:
        raise ValueError(f"Failed to infer n_observables from phi_func: {e}")

    logger.info(f"Number of Koopman observables: {n_observables}")
    C_matrix = np.zeros((system.n_state_obs, n_observables))
    if system.n_state_obs <= n_observables:
        C_matrix[:, :system.n_state_obs] = np.eye(system.n_state_obs)
    else:
        logger.warning("n_state_obs > n_observables, default C matrix may be problematic.")
        C_matrix[:n_observables, :n_observables] = np.eye(n_observables)
    koopman_learner = Koopman_SysID(
        n_state=system.n_state_obs,
        n_control=system.n_control,
        n_observables=n_observables,
        phi_func=phi_func,
        C_matrix=C_matrix
    )
    koopman_learner.fit(state_traj_list_obs, control_traj_list, lambda_reg=1e-4)
    A_np, B_np, C_np, phi_fn_np, n_lifted = koopman_learner.get_model()
    if A_np is None:
        raise RuntimeError("Koopman model fitting failed.")
    logger.info("Koopman model identified successfully.")
    return A_np, B_np, C_np, phi_fn_np, n_lifted


# ==============================================================================
# MPC Controller Definition using OCSys
# ==============================================================================
class KoopmanMPC:
    """
    MPC controller built on top of a learned Koopman model.
    """
    def __init__(self, system_params, koopman_model, mpc_horizon, initial_theta_cost_np, control_limits=None):
        self.A_k, self.B_k, self.C_k, self.phi_k, self.n_k_lifted = koopman_model
        self.n_state_obs = system_params['n_state_obs']
        self.n_control = system_params['n_control']
        self.dt = system_params['dt']
        self.mpc_horizon = mpc_horizon

        self.ocs = OCSys(project_name="Koopman_MPC_OCP")
        self.x_obs_sym_ocs = SX.sym('x_obs_ocs', self.n_state_obs)
        self.ocs.setStateVariable(self.x_obs_sym_ocs)

        self.u_sym_ocs = SX.sym('u_ocs', self.n_control)

        if control_limits is None:
            control_lb = [-3.0] * self.n_control
            control_ub = [3.0] * self.n_control
        else:
            control_lb = control_limits[0]
            control_ub = control_limits[1]

        self.ocs.setControlVariable(
            self.u_sym_ocs,
            control_lb=control_lb,
            control_ub=control_ub
        )

        self.n_theta_cost = self.n_state_obs + self.n_control
        self.theta_cost_sym_ocs = SX.sym('theta_cost_ocs', self.n_theta_cost)
        self.ocs.setAuxvarVariable(self.theta_cost_sym_ocs)
        self.current_theta_cost_np = initial_theta_cost_np.copy()
        self.theta_history = [initial_theta_cost_np.copy()]

        self.ocs.setKoopmanModel(self.A_k, self.B_k, self.C_k, self.phi_k, self.n_k_lifted)
        self.ocs.setDyn()

        def mpc_path_cost_lambda(x_obs_lambda, u_lambda, theta_cost_lambda):
            Q_diag_elements = [exp(theta_cost_lambda[i]) for i in range(self.n_state_obs)]
            R_diag_elements = [exp(theta_cost_lambda[i + self.n_state_obs]) for i in range(self.n_control)]
            Q_mat = diag(vertcat(*Q_diag_elements))
            R_mat = diag(vertcat(*R_diag_elements))
            target_x_obs = SX.zeros(self.n_state_obs)
            error_x = x_obs_lambda - target_x_obs
            cost = 0.5 * mtimes([error_x.T, Q_mat, error_x]) + 0.5 * mtimes([u_lambda.T, R_mat, u_lambda])
            return cost

        def mpc_final_cost_lambda(x_obs_lambda, theta_cost_lambda):
            Q_diag_elements_final = [exp(theta_cost_lambda[i]) * 10.0 for i in range(self.n_state_obs)]
            Q_mat_final = diag(vertcat(*Q_diag_elements_final))
            target_x_obs = SX.zeros(self.n_state_obs)
            error_x = x_obs_lambda - target_x_obs
            return 0.5 * mtimes([error_x.T, Q_mat_final, error_x])

        self.ocs.setPathCost(mpc_path_cost_lambda)
        self.ocs.setFinalCost(mpc_final_cost_lambda)
        logger.info(f"KoopmanMPC initialised with {self.n_theta_cost} cost parameters.")

    def solve_mpc_step(self, current_obs_state_np, current_theta_cost_np_val):
        """
        Solve the MPC problem for a single time step.
        """
        if current_theta_cost_np_val.ndim == 1:
            current_theta_cost_np_val = current_theta_cost_np_val.reshape(-1)
        mpc_sol = self.ocs.ocSolver(
            ini_state_orig=current_obs_state_np,
            horizon=self.mpc_horizon,
            auxvar_value=current_theta_cost_np_val,
            print_level=0,
            costate_option=0
        )
        if mpc_sol["success"] and mpc_sol["control_traj_opt"] is not None and len(mpc_sol["control_traj_opt"]) > 0:
            u_mpc_step = mpc_sol["control_traj_opt"][0, :]
            return u_mpc_step, mpc_sol
        else:
            logger.warning(f"MPC step failed to solve: {mpc_sol.get('solver_stats', {}).get('return_status', 'N/A')}. Returning zero control.")
            return np.zeros(self.n_control), mpc_sol

    def update_theta_cost(self, new_theta_cost_np):
        self.current_theta_cost_np = new_theta_cost_np.copy()
        self.theta_history.append(new_theta_cost_np.copy())


# ==============================================================================
# Meta-Learning Loop for MPC Cost Parameters using PDP
# ==============================================================================
def learn_mpc_cost_params_pdp(
    true_system,
    koopman_mpc_controller,
    sim_horizon_meta,
    num_meta_iterations,
    learning_rate_meta,
    initial_true_state_full_np
):
    """
    Perform meta-learning on the MPC cost parameters using PDP.
    """
    logger.info("Starting meta-learning of MPC cost parameters using PDP...")
    meta_cost_history = []
    theta_cost_current_np = koopman_mpc_controller.current_theta_cost_np.copy()
    theta_history_list = [theta_cost_current_np.copy()]

    aux_lqr_solver = LQR(project_name="PDP_Aux_LQR_for_MPC")
    for meta_iter in range(num_meta_iterations):
        time_meta_iter_start = time.time()
        x_full_k_true = initial_true_state_full_np.copy()
        # h_target_episode = 0.0 # Removed specific logic
        episode_x_obs_true_list = []
        episode_u_mpc_list = []
        episode_du_dtheta_cost_list = []
        meta_episode_cost = 0.0
        for t_meta in range(sim_horizon_meta):
            # Assume observed state is the full state or subset provided by system
            current_x_obs_true = x_full_k_true[:true_system.n_state_obs]
            episode_x_obs_true_list.append(current_x_obs_true.copy())

            u_mpc_t, mpc_ocp_solution = koopman_mpc_controller.solve_mpc_step(
                current_x_obs_true, theta_cost_current_np
            )
            episode_u_mpc_list.append(u_mpc_t.copy())

            if mpc_ocp_solution["success"] and mpc_ocp_solution["lifted_state_traj_opt"] is not None and \
               mpc_ocp_solution["control_traj_opt"] is not None and mpc_ocp_solution["costate_traj_opt"] is not None:
                if not koopman_mpc_controller.ocs._pmp_diff_done:
                    koopman_mpc_controller.ocs.diffPMP()
                aux_sys_coeffs = koopman_mpc_controller.ocs.getAuxSys(
                    state_traj_opt=mpc_ocp_solution["lifted_state_traj_opt"],
                    control_traj_opt=mpc_ocp_solution["control_traj_opt"],
                    costate_traj_opt=mpc_ocp_solution["costate_traj_opt"],
                    auxvar_value=theta_cost_current_np
                )
                if aux_sys_coeffs:
                    try:
                        aux_lqr_solver.setDyn(aux_sys_coeffs['dynF'], aux_sys_coeffs['dynG'], aux_sys_coeffs['dynE'])
                        aux_lqr_solver.setPathCost(aux_sys_coeffs['Hxx'], aux_sys_coeffs['Huu'],
                                                   Hxu=aux_sys_coeffs['Hxu'], Hux=aux_sys_coeffs['Hux'],
                                                   Hxe=aux_sys_coeffs['Hxe'], Hue=aux_sys_coeffs['Hue'])
                        aux_lqr_solver.setFinalCost(aux_sys_coeffs['hxx'], aux_sys_coeffs['hxe'])
                        ini_X_aux = np.zeros((koopman_mpc_controller.n_k_lifted, koopman_mpc_controller.n_theta_cost))
                        aux_lqr_sol = aux_lqr_solver.lqrSolver(ini_X_aux, koopman_mpc_controller.mpc_horizon)
                        if aux_lqr_sol['control_traj_opt'] is not None and len(aux_lqr_sol['control_traj_opt']) > 0:
                            du_dtheta_t = aux_lqr_sol['control_traj_opt'][0]
                            episode_du_dtheta_cost_list.append(du_dtheta_t.copy())
                        else:
                            logger.warning(f"Aux LQR solve failed at t_meta={t_meta}. Using zero gradient.")
                            episode_du_dtheta_cost_list.append(
                                np.zeros((koopman_mpc_controller.n_control, koopman_mpc_controller.n_theta_cost))
                            )
                    except Exception as e_lqr:
                        logger.error(f"Error setting up or solving Aux LQR at t_meta={t_meta}: {e_lqr}")
                        episode_du_dtheta_cost_list.append(
                            np.zeros((koopman_mpc_controller.n_control, koopman_mpc_controller.n_theta_cost))
                        )
                else:
                    logger.warning(f"getAuxSys failed at t_meta={t_meta}. Using zero gradient.")
                    episode_du_dtheta_cost_list.append(
                        np.zeros((koopman_mpc_controller.n_control, koopman_mpc_controller.n_theta_cost))
                    )
            else:
                logger.warning(f"MPC failed at t_meta={t_meta}. Using zero gradient.")
                episode_du_dtheta_cost_list.append(
                    np.zeros((koopman_mpc_controller.n_control, koopman_mpc_controller.n_theta_cost))
                )

            # Step true system
            x_full_k_true = true_system.dynamics_step(x_full_k_true, u_mpc_t)

            # Simple meta stage cost: squared distance to origin for first state + small control penalty
            # This assumes state[0] is something we want to regulate.
            # Adapted to be generic if possible.
            p1_true = x_full_k_true[0]
            # p2_true = x_full_k_true[2]
            target_p1_meta = 0.0
            meta_stage_cost = (p1_true - target_p1_meta) ** 2 + 0.01 * np.sum(u_mpc_t ** 2)
            meta_episode_cost += meta_stage_cost

        meta_cost_history.append(meta_episode_cost)

        grad_L_meta_theta_np = np.zeros_like(theta_cost_current_np, dtype=float)
        if len(episode_du_dtheta_cost_list) == sim_horizon_meta:
            for t_meta in range(sim_horizon_meta):
                u_mpc_t_val = episode_u_mpc_list[t_meta]
                du_dtheta_t_val = episode_du_dtheta_cost_list[t_meta]
                dl_meta_du_t = 0.02 * u_mpc_t_val
                grad_summand_meta = du_dtheta_t_val.T @ dl_meta_du_t.reshape(-1, 1)
                grad_L_meta_theta_np += grad_summand_meta.flatten()
        else:
            logger.warning("Mismatch in du_dtheta list length, skipping gradient update for this meta-iteration.")

        grad_norm = np.linalg.norm(grad_L_meta_theta_np)
        if grad_norm > 10.0:
            grad_L_meta_theta_np = (grad_L_meta_theta_np / grad_norm) * 10.0
            logger.info(f"MetaIter: {meta_iter + 1} - Gradient clipped from {grad_norm:.2e}")

        theta_cost_current_np -= learning_rate_meta * grad_L_meta_theta_np
        koopman_mpc_controller.update_theta_cost(theta_cost_current_np)
        theta_history_list.append(theta_cost_current_np.copy())

        time_meta_iter_end = time.time()
        actual_weights_str = ", ".join([f"{w:.3f}" for w in np.exp(theta_cost_current_np)])
        logger.info(
            f"MetaIter: {meta_iter + 1}/{num_meta_iterations}, MetaCost: {meta_episode_cost:.3e}, "
            f"GradNorm: {grad_norm:.2e}, ActualWeights: [{actual_weights_str}], "
            f"Time: {time_meta_iter_end - time_meta_iter_start:.2f}s"
        )
    logger.info("Meta-learning of MPC cost parameters finished.")
    return koopman_mpc_controller.current_theta_cost_np, meta_cost_history, theta_history_list
