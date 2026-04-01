
import do_mpc
from do_mpc.approximateMPC import ApproxMPC, AMPCSampler, Trainer
import torch
import numpy as np
import matplotlib.pyplot as plt

class CustomAMPCSampler(AMPCSampler):
    def setup_simulator(self):
        self.simulator = do_mpc.simulator.Simulator(self.mpc.model)
        self.simulator.settings.t_step = self.mpc.settings.t_step

        def p_fun_adapt(t_now):
            p_mpc = self.mpc.p_fun(t_now)
            p_sim = self.simulator.get_p_template()

            # Iterate over simulator parameters and try to fetch from mpc parameters (scenario 0)
            for key in p_sim.keys():
                try:
                    # Scenario based access
                    p_sim[key] = p_mpc['_p', 0, key]
                except:
                    try:
                        # Direct access
                        p_sim[key] = p_mpc[key]
                    except:
                        pass
            return p_sim

        self.simulator.set_p_fun(p_fun_adapt)
        self.simulator.setup()

class PDP:
    def __init__(self, mpc, simulator=None, estimator=None):
        self.mpc = mpc
        self.simulator = simulator
        self.estimator = estimator
        self.approx_mpc = ApproxMPC(mpc)
        self.sampler = CustomAMPCSampler(mpc)
        self.trainer = Trainer(self.approx_mpc)
        self.model = mpc.model

    def setup_approx_mpc(self, n_hidden_layers=1, n_neurons=50):
        self.approx_mpc.settings.n_hidden_layers = n_hidden_layers
        self.approx_mpc.settings.n_neurons = n_neurons
        self.approx_mpc.setup()

    def setup_sampler(self, dataset_name='my_dataset', n_samples=1000, trajectory_length=1, closed_loop_flag=True):
        self.sampler.settings.dataset_name = dataset_name
        self.sampler.settings.n_samples = n_samples
        self.sampler.settings.trajectory_length = trajectory_length
        self.sampler.settings.closed_loop_flag = closed_loop_flag
        self.sampler.setup()

    def setup_trainer(self, dataset_name='my_dataset', n_epochs=3000, save_fig=True, save_history=True, show_fig=False):
        self.trainer.settings.dataset_name = dataset_name
        self.trainer.settings.n_epochs = n_epochs
        self.trainer.settings.save_fig = save_fig
        self.trainer.settings.save_history = save_history
        self.trainer.settings.show_fig = show_fig
        self.trainer.settings.scheduler_flag = True
        self.trainer.scheduler_settings.cooldown = 0
        self.trainer.scheduler_settings.patience = 50
        self.trainer.setup()

    def generate_data(self, seed=42):
        np.random.seed(seed)
        self.sampler.default_sampling()

    def train(self, seed=42):
        torch.manual_seed(seed)
        self.trainer.default_training()

    def get_controller(self):
        return self.approx_mpc

    def run_simulation(self, x0, sim_time=100, show_animation=False, graphics=None, store_results=False):
        if self.simulator is None:
            raise ValueError("Simulator not provided.")

        self.simulator.x0 = x0
        self.approx_mpc.x0 = x0
        if self.estimator:
            self.estimator.x0 = x0

        # self.approx_mpc.set_initial_guess()
        self.simulator.reset_history()
        # self.approx_mpc.reset_history()

        for k in range(sim_time):
            u0 = self.approx_mpc.make_step(x0, clip_to_bounds=True)
            y_next = self.simulator.make_step(u0)

            if self.estimator:
                x0 = self.estimator.make_step(y_next)
            else:
                x0 = y_next

            if show_animation and graphics:
                graphics.plot_results(t_ind=k)
                graphics.reset_axes()
                plt.show()
                plt.pause(0.01)

        if store_results:
             do_mpc.data.save_results([self.simulator], "PDP_MPC")

        return self.simulator.data
