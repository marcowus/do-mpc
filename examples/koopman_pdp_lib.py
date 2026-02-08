"""
This file implements a complete example of differentiable cost learning for a
non-linear dynamical system using a Koopman operator based model predictive
controller (MPC) and a Pontryagin based parametric differentiation procedure
(PDP).
"""

from casadi import *  # CasADi symbols, functions and algebraic operations
import numpy as np
import logging
import time
import os
import matplotlib.pyplot as plt

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
    def __init__(self, n_state, n_control, n_observables, phi_func,
                 C_matrix=None, project_name='Koopman EDMDc'):
        self.project_name = project_name
        self.n_state = n_state
        self.n_control = n_control
        self.n_observables = n_observables
        self.phi_func = phi_func

        if C_matrix is None:
            logger.info("No C_matrix provided, assuming C = [I | 0].")
            self.C = np.zeros((n_state, n_observables))
            if n_state <= n_observables:
                self.C[:, :n_state] = np.eye(n_state)
            else:
                logger.error("Cannot construct default C when n_state > n_observables.")
                raise ValueError("n_state cannot exceed n_observables when C_matrix is not provided.")
        else:
            if not isinstance(C_matrix, np.ndarray):
                raise TypeError("C_matrix must be a numpy array.")
            self.C = C_matrix

        self.A = None
        self.B = None
        self.is_fitted_ = False

        try:
            test_x = np.zeros(self.n_state)
            test_z = self.phi_func(test_x)
            if test_z.shape != (self.n_observables,):
                raise ValueError(f"phi_func output dimension {test_z.shape} != {self.n_observables}")
        except Exception as e:
            raise ValueError(f"Error testing phi_func: {e}")

    def fit(self, state_traj_list, control_traj_list, lambda_reg=1e-6):
        Z_curr, Z_next, U_curr = [], [], []
        total_points = 0
        num_traj = len(state_traj_list)

        for i in range(num_traj):
            states = state_traj_list[i]
            controls = control_traj_list[i]
            if np.any(np.isnan(states)) or np.any(np.isnan(controls)) or \
               np.any(np.isinf(states)) or np.any(np.isinf(controls)):
                continue
            if states.shape[0] != controls.shape[0] + 1: continue

            horizon = controls.shape[0]
            for t in range(horizon):
                try:
                    x_k = states[t, :]
                    x_k_next = states[t + 1, :]
                    u_k = controls[t, :]
                    z_k = self.phi_func(x_k)
                    z_k_next = self.phi_func(x_k_next)
                    Z_curr.append(z_k)
                    Z_next.append(z_k_next)
                    U_curr.append(u_k)
                    total_points += 1
                except: continue

        if total_points == 0: raise RuntimeError("No valid data points.")

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
            AB = ZG @ np.linalg.pinv(GGT)

        self.A = AB[:, :self.n_observables]
        self.B = AB[:, self.n_observables:]
        self.is_fitted_ = True
        logger.info("Koopman model (A, B) fitted successfully.")
        return self

    def predict(self, ini_state_orig, control_seq):
        if not self.is_fitted_: return None
        horizon = control_seq.shape[0]
        pred_state_traj_orig = np.zeros((horizon + 1, self.n_state))
        pred_state_traj_orig[0, :] = ini_state_orig.flatten()
        try:
            z_k = self.phi_func(ini_state_orig.flatten())
        except: return None

        for t in range(horizon):
            try:
                u_k = control_seq[t, :]
                z_k_next = self.A @ z_k + self.B @ u_k
                x_k_next = self.C @ z_k_next
                pred_state_traj_orig[t + 1, :] = x_k_next
                z_k = z_k_next
            except:
                pred_state_traj_orig[t + 1:, :] = np.nan
                break
        return pred_state_traj_orig

    def get_model(self):
        if not self.is_fitted_: return None, None, self.C, self.phi_func, self.n_observables
        return self.A, self.B, self.C, self.phi_func, self.n_observables


class OCSys:
    def __init__(self, project_name="OCSys"):
        self.state_orig_sym = None
        self.control_sym = None
        self.auxvar = None
        self.state = None
        self.control = None
        self.using_koopman = False
        self._pmp_diff_done = False
        # Attributes...
        self.n_state = 0
        self.n_control = 0
        self.n_auxvar = 0
        self.control_lb = []
        self.control_ub = []
        self.state_lb = []
        self.state_ub = []
        self.dyn_fn = None
        self.path_cost_fn = None
        self.final_cost_fn = None
        self.A_casadi = None
        self.B_casadi = None
        self.C_casadi = None

        # Derivatives functions
        self.dfx_fn = None
        self.dfu_fn = None
        self.dfe_fn = None
        self.ddHxx_fn = None
        self.ddHxu_fn = None
        self.ddHxe_fn = None
        self.ddHux_fn = None
        self.ddHuu_fn = None
        self.ddHue_fn = None
        self.ddhxx_fn = None
        self.ddhxe_fn = None

    def setAuxvarVariable(self, auxvar_sym):
        self.auxvar = auxvar_sym if auxvar_sym is not None else SX.sym('p_dummy')
        self.n_auxvar = self.auxvar.numel()
        self._pmp_diff_done = False

    def setStateVariable(self, state_orig_sym):
        self.state_orig_sym = state_orig_sym
        self.n_state_orig = self.state_orig_sym.numel()
        self.state = self.state_orig_sym
        self.n_state = self.n_state_orig
        self.state_lb = [-1e20]*self.n_state
        self.state_ub = [1e20]*self.n_state
        self._pmp_diff_done = False

    def setControlVariable(self, control_sym, control_lb=None, control_ub=None):
        self.control = control_sym
        self.n_control = self.control.numel()
        self.control_lb = list(control_lb) if control_lb else [-1e20]*self.n_control
        self.control_ub = list(control_ub) if control_ub else [1e20]*self.n_control
        self._pmp_diff_done = False

    def setKoopmanModel(self, A_np, B_np, C_np, phi_func_np, n_lifted):
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
        self.state_lb = [-1e20]*self.n_state
        self.state_ub = [1e20]*self.n_state
        self.using_koopman = True
        self._pmp_diff_done = False

    def setDyn(self, ode_expr=None):
        if self.using_koopman:
            self.dyn = mtimes(self.A_casadi, self.state) + mtimes(self.B_casadi, self.control)
            self.dyn_fn = Function('k_dyn', [self.state, self.control, self.auxvar], [self.dyn])
        else:
            self.dyn = ode_expr
            self.dyn_fn = Function('dyn', [self.state, self.control, self.auxvar], [self.dyn])
        self._pmp_diff_done = False

    def setPathCost(self, path_cost_lambda):
        self.path_cost, self.path_cost_fn = self._create_cost_func(path_cost_lambda, "path")
        self._pmp_diff_done = False

    def setFinalCost(self, final_cost_lambda):
        self.final_cost, self.final_cost_fn = self._create_cost_func(final_cost_lambda, "final")
        self._pmp_diff_done = False

    def _create_cost_func(self, cost_lambda, cost_type):
        args = [self.state_orig_sym, self.control, self.auxvar] if cost_type == "path" else [self.state_orig_sym, self.auxvar]
        cost_expr_orig = cost_lambda(*args)
        if self.using_koopman:
            x_recon = mtimes(self.C_casadi, self.state)
            cost_expr = substitute(cost_expr_orig, self.state_orig_sym, x_recon)
        else:
            cost_expr = cost_expr_orig
        input_vars = [self.state, self.control, self.auxvar] if cost_type == "path" else [self.state, self.auxvar]
        return cost_expr, Function(f"{cost_type}_cost", input_vars, [cost_expr])

    def ocSolver(self, ini_state_orig, horizon, auxvar_value=None, print_level=0, costate_option=0):
        if isinstance(ini_state_orig, (list, np.ndarray)): ini_state_orig_np = np.array(ini_state_orig).flatten()
        if auxvar_value is None: auxvar_value_np = np.array([])
        else: auxvar_value_np = np.array(auxvar_value).flatten()

        if self.using_koopman:
            ini_state_nlp = self.phi_func_np(ini_state_orig_np).tolist()
            n_st = self.n_lifted_state
        else:
            ini_state_nlp = ini_state_orig_np.tolist()
            n_st = self.n_state

        w, w0, lbw, ubw, g, lbg, ubg = [], [], [], [], [], [], []
        Xk = MX.sym('X0', n_st)
        w += [Xk]; lbw += ini_state_nlp; ubw += ini_state_nlp; w0 += ini_state_nlp
        J = 0

        for k in range(horizon):
            Uk = MX.sym(f'U_{k}', self.n_control)
            w += [Uk]; lbw += self.control_lb; ubw += self.control_ub; w0 += [0.0]*self.n_control
            Xk_next = self.dyn_fn(Xk, Uk, auxvar_value_np)
            J += self.path_cost_fn(Xk, Uk, auxvar_value_np)
            Xk_var = MX.sym(f'X_{k+1}', n_st)
            w += [Xk_var]; lbw += [-1e20]*n_st; ubw += [1e20]*n_st; w0 += ini_state_nlp
            g += [Xk_next - Xk_var]; lbg += [0]*n_st; ubg += [0]*n_st
            Xk = Xk_var
        J += self.final_cost_fn(Xk, auxvar_value_np)

        solver = nlpsol('solver', 'ipopt', {'f':J, 'x':vertcat(*w), 'g':vertcat(*g)},
                        {'ipopt.print_level':print_level, 'ipopt.sb':'yes', 'print_time':False})
        try:
            sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        except: return {"success": False}

        success = solver.stats()['success']
        if not success: return {"success": False}

        w_opt = sol['x'].full().flatten()
        ctrl = []
        state_lift = [w_opt[:n_st]]
        idx = n_st
        for _ in range(horizon):
            ctrl.append(w_opt[idx:idx+self.n_control])
            idx += self.n_control
            state_lift.append(w_opt[idx:idx+n_st])
            idx += n_st

        opt_sol = {
            "control_traj_opt": np.array(ctrl),
            "costate_traj_opt": np.reshape(sol['lam_g'].full(), (horizon, n_st)),
            "success": True,
            "lifted_state_traj_opt": np.array(state_lift) if self.using_koopman else None
        }
        if self.using_koopman:
            opt_sol["state_traj_opt"] = (self.C_np @ opt_sol["lifted_state_traj_opt"].T).T
        else:
            opt_sol["state_traj_opt"] = opt_sol["lifted_state_traj_opt"] # Not strictly correct logic but simplified

        return opt_sol

    def diffPMP(self):
        # Same logic as before for differentiation
        self.costate = SX.sym('lambda', self.n_state)
        self.path_Hamil = self.path_cost + dot(self.costate, self.dyn)
        self.final_Hamil = self.final_cost
        # Derivatives...
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfe_fn = Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])

        self.dHx = jacobian(self.path_Hamil, self.state).T
        self.dHx_fn = Function('dHx', [self.state, self.control, self.costate, self.auxvar], [self.dHx])
        self.dHu = jacobian(self.path_Hamil, self.control).T
        self.dHu_fn = Function('dHu', [self.state, self.control, self.costate, self.auxvar], [self.dHu])

        self.ddHxx = jacobian(self.dHx, self.state)
        self.ddHxx_fn = Function('ddHxx', [self.state, self.control, self.costate, self.auxvar], [self.ddHxx])
        self.ddHxu = jacobian(self.dHx, self.control)
        self.ddHxu_fn = Function('ddHxu', [self.state, self.control, self.costate, self.auxvar], [self.ddHxu])
        self.ddHxe = jacobian(self.dHx, self.auxvar)
        self.ddHxe_fn = Function('ddHxe', [self.state, self.control, self.costate, self.auxvar], [self.ddHxe])
        self.ddHux = jacobian(self.dHu, self.state)
        self.ddHux_fn = Function('ddHux', [self.state, self.control, self.costate, self.auxvar], [self.ddHux])
        self.ddHuu = jacobian(self.dHu, self.control)
        self.ddHuu_fn = Function('ddHuu', [self.state, self.control, self.costate, self.auxvar], [self.ddHuu])
        self.ddHue = jacobian(self.dHu, self.auxvar)
        self.ddHue_fn = Function('ddHue', [self.state, self.control, self.costate, self.auxvar], [self.ddHue])

        self.dhx = jacobian(self.final_Hamil, self.state).T
        self.dhx_fn = Function('dhx', [self.state, self.auxvar], [self.dhx])
        self.ddhxx = jacobian(self.dhx, self.state)
        self.ddhxx_fn = Function('ddhxx', [self.state, self.auxvar], [self.ddhxx])
        self.ddhxe = jacobian(self.dhx, self.auxvar)
        self.ddhxe_fn = Function('ddhxe', [self.state, self.auxvar], [self.ddhxe])
        self._pmp_diff_done = True

    def getAuxSys(self, state_traj, control_traj, costate_traj, auxvar_value):
        # Simplified aux sys retrieval
        horizon = len(control_traj)
        aux = auxvar_value
        dynF, dynG, dynE = [], [], []
        Hxx, Hxu, Hxe, Hux, Huu, Hue = [], [], [], [], [], []

        for k in range(horizon):
            st, ct, cst = state_traj[k], control_traj[k], costate_traj[k]
            dynF.append(self.dfx_fn(st, ct, aux).full())
            dynG.append(self.dfu_fn(st, ct, aux).full())
            dynE.append(self.dfe_fn(st, ct, aux).full())
            Hxx.append(self.ddHxx_fn(st, ct, cst, aux).full())
            Hxu.append(self.ddHxu_fn(st, ct, cst, aux).full())
            Hxe.append(self.ddHxe_fn(st, ct, cst, aux).full())
            Hux.append(self.ddHux_fn(st, ct, cst, aux).full())
            Huu.append(self.ddHuu_fn(st, ct, cst, aux).full())
            Hue.append(self.ddHue_fn(st, ct, cst, aux).full())

        st_final = state_traj[-1]
        hxx = [self.ddhxx_fn(st_final, aux).full()]
        hxe = [self.ddhxe_fn(st_final, aux).full()]

        return {"dynF": dynF, "dynG": dynG, "dynE": dynE, "Hxx": Hxx, "Huu": Huu, "Hxu": Hxu, "Hux": Hux, "Hxe": Hxe, "Hue": Hue, "hxx": hxx, "hxe": hxe}


class LQR:
    def __init__(self, project_name="LQR"):
        self.dynF = None
    def setDyn(self, F, G, E):
        self.dynF, self.dynG, self.dynE = F, G, E
        self.n_state = F[0].shape[0]
        self.horizon = len(F)
    def setPathCost(self, Hxx, Huu, Hxu, Hux, Hxe, Hue):
        self.Hxx, self.Huu, self.Hxu, self.Hux, self.Hxe, self.Hue = Hxx, Huu, Hxu, Hux, Hxe, Hue
    def setFinalCost(self, hxx, hxe):
        self.hxx, self.hxe = hxx, hxe
    def lqrSolver(self, ini_state, horizon):
        P = [None]*(horizon+1)
        W = [None]*(horizon+1)
        P[horizon] = self.hxx[0]
        W[horizon] = self.hxe[0]
        K, k_ff = [None]*horizon, [None]*horizon

        for t in range(horizon-1, -1, -1):
            Q_uu = self.Huu[t] + self.dynG[t].T @ P[t+1] @ self.dynG[t]
            Q_ux = self.Hux[t] + self.dynG[t].T @ P[t+1] @ self.dynF[t]
            Q_u = self.Hue[t] + self.dynG[t].T @ W[t+1] + self.dynG[t].T @ P[t+1] @ self.dynE[t]
            Q_xx = self.Hxx[t] + self.dynF[t].T @ P[t+1] @ self.dynF[t]
            Q_x = self.Hxe[t] + self.dynF[t].T @ W[t+1] + self.dynF[t].T @ P[t+1] @ self.dynE[t]

            try: inv_Quu = np.linalg.inv(Q_uu)
            except: inv_Quu = np.linalg.pinv(Q_uu)

            K[t] = -inv_Quu @ Q_ux
            k_ff[t] = -inv_Quu @ Q_u
            P[t] = Q_xx + Q_ux.T @ K[t]
            W[t] = Q_x + Q_ux.T @ k_ff[t]

        x_traj = [ini_state.flatten().reshape(-1,1)]
        u_traj = []
        for t in range(horizon):
            xt = x_traj[-1]
            ut = K[t] @ xt + k_ff[t]
            xt_next = self.dynF[t] @ xt + self.dynG[t] @ ut + self.dynE[t]
            x_traj.append(xt_next)
            u_traj.append(ut)

        return {"control_traj_opt": [u.flatten() for u in u_traj]}


def identify_koopman_model(system, state_traj_list_obs, control_traj_list, phi_func=None):
    if phi_func is None:
        def default_phi_func(x):
            poly = [x]
            n = len(x)
            for i in range(n):
                for j in range(i, n):
                    poly.append(x[i]*x[j])
            return np.hstack(poly)
        phi_func = default_phi_func

    n_obs = phi_func(np.zeros(system.n_state_obs)).shape[0]
    kid = Koopman_SysID(system.n_state_obs, system.n_control, n_obs, phi_func)
    kid.fit(state_traj_list_obs, control_traj_list)
    return kid.get_model()


class KoopmanMPC:
    def __init__(self, system_params, koopman_model, mpc_horizon, initial_theta_cost_np, control_limits=None, target_state=None):
        self.A_k, self.B_k, self.C_k, self.phi_k, self.n_k_lifted = koopman_model
        self.n_state_obs = system_params['n_state_obs']
        self.n_control = system_params['n_control']
        self.mpc_horizon = mpc_horizon

        self.target_state_np = np.zeros(self.n_state_obs) if target_state is None else np.array(target_state).flatten()
        logger.info(f"KoopmanMPC initialised with target: {self.target_state_np}")

        self.ocs = OCSys("KoopmanMPC")
        self.x_obs_sym = SX.sym('x', self.n_state_obs)
        self.ocs.setStateVariable(self.x_obs_sym)
        self.u_sym = SX.sym('u', self.n_control)

        clb = control_limits[0] if control_limits else [-3.0]*self.n_control
        cub = control_limits[1] if control_limits else [3.0]*self.n_control
        self.ocs.setControlVariable(self.u_sym, clb, cub)

        self.n_theta = self.n_state_obs + self.n_control
        self.theta_sym = SX.sym('theta', self.n_theta)
        self.ocs.setAuxvarVariable(self.theta_sym)

        self.ocs.setKoopmanModel(self.A_k, self.B_k, self.C_k, self.phi_k, self.n_k_lifted)
        self.ocs.setDyn()

        self.current_theta = initial_theta_cost_np.copy()

        def path_cost(x, u, theta):
            Q = diag(vertcat(*[exp(theta[i]) for i in range(self.n_state_obs)]))
            R = diag(vertcat(*[exp(theta[i+self.n_state_obs]) for i in range(self.n_control)]))
            err = x - SX(self.target_state_np)
            return 0.5 * mtimes([err.T, Q, err]) + 0.5 * mtimes([u.T, R, u])

        def final_cost(x, theta):
            Q = diag(vertcat(*[exp(theta[i])*10.0 for i in range(self.n_state_obs)]))
            err = x - SX(self.target_state_np)
            return 0.5 * mtimes([err.T, Q, err])

        self.ocs.setPathCost(path_cost)
        self.ocs.setFinalCost(final_cost)

    def solve_mpc_step(self, current_x, theta):
        return self.ocs.ocSolver(current_x, self.mpc_horizon, theta)

    def update_theta_cost(self, theta):
        self.current_theta = theta.copy()


def learn_mpc_cost_params_pdp(true_system, koopman_mpc_controller, sim_horizon_meta, num_meta_iterations, learning_rate_meta, initial_true_state_full_np, target_state_meta=None):
    logger.info("Starting meta-learning...")
    theta = koopman_mpc_controller.current_theta.copy()
    meta_hist, theta_hist = [], [theta.copy()]
    aux_lqr = LQR()

    tgt = np.zeros(true_system.n_state_obs) if target_state_meta is None else np.array(target_state_meta).flatten()

    for itr in range(num_meta_iterations):
        x = initial_true_state_full_np.copy()
        ep_u, ep_du, cost = [], [], 0

        for t in range(sim_horizon_meta):
            res = koopman_mpc_controller.solve_mpc_step(x, theta)
            if not res['success']:
                ep_du.append(np.zeros((true_system.n_control, len(theta))))
                ep_u.append(np.zeros(true_system.n_control))
                continue

            u = res['control_traj_opt'][0]
            ep_u.append(u)

            aux = koopman_mpc_controller.ocs.getAuxSys(res['lifted_state_traj_opt'], res['control_traj_opt'], res['costate_traj_opt'], theta)
            aux_lqr.setDyn(aux['dynF'], aux['dynG'], aux['dynE'])
            aux_lqr.setPathCost(aux['Hxx'], aux['Huu'], aux['Hxu'], aux['Hux'], aux['Hxe'], aux['Hue'])
            aux_lqr.setFinalCost(aux['hxx'], aux['hxe'])
            sol_aux = aux_lqr.lqrSolver(np.zeros(koopman_mpc_controller.n_k_lifted), koopman_mpc_controller.mpc_horizon)
            ep_du.append(np.array(sol_aux['control_traj_opt'][0]).reshape(true_system.n_control, -1))

            x = true_system.dynamics_step(x, u)
            cost += np.sum((x - tgt)**2) + 0.01*np.sum(u**2)

        meta_hist.append(cost)
        grad = np.zeros_like(theta)
        for t in range(len(ep_u)):
            grad += ep_du[t].T @ (0.02 * ep_u[t])

        gn = np.linalg.norm(grad)
        if gn > 100.0: grad = grad/gn * 100.0
        theta -= learning_rate_meta * grad
        koopman_mpc_controller.update_theta_cost(theta)
        theta_hist.append(theta.copy())

        logger.info(f"MetaIter: {itr+1}/{num_meta_iterations}, MetaCost: {cost:.3e}, GradNorm: {gn:.2e}")

    return theta, meta_hist, theta_hist

def plot_results(true_sys, history_x, history_u, title="Results"):
    history_x = np.array(history_x)
    history_u = np.array(history_u)
    t = np.arange(len(history_x))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(t, history_x)
    ax[0].set_title(f"{title} - States")
    ax[0].grid(True)

    # Adjust control length if needed
    if len(history_u) > 0:
        t_u = np.arange(len(history_u))
        ax[1].plot(t_u, history_u)

    ax[1].set_title(f"{title} - Controls")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"results_{title.replace(' ', '_')}.png")
    logger.info(f"Saved plot to results_{title.replace(' ', '_')}.png")
