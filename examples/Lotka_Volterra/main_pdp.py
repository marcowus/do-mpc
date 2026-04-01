
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

# Add examples/ to path to import PDP
sys.path.append(os.path.join('..'))
from PDP import PDP

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

def main():
    # Model setup
    model = template_model()
    mpc = template_mpc(model)
    simulator = template_simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)

    # Initial conditions
    x0 = np.array([0.5, 0.7]).reshape(-1, 1)
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    # Initialize PDP
    pdp = PDP(mpc, simulator, estimator)

    # Setup
    pdp.setup_approx_mpc(n_hidden_layers=1, n_neurons=20)
    # Using small number of samples and epochs for quick verification
    pdp.setup_sampler(n_samples=50, trajectory_length=1)
    pdp.setup_trainer(n_epochs=2, show_fig=False)

    # Generate data and train
    print("Generating data...")
    pdp.generate_data()
    print("Training...")
    pdp.train()

    # Run simulation
    print("Running simulation...")
    # Setup graphics
    # Note: do_mpc.graphics.default_plot returns fig, ax, graphics
    # We initialize it with simulator.data to visualize the simulation results
    fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data, figsize=(8,5))

    # Run simulation using the trained approximate controller
    pdp.run_simulation(x0, sim_time=20, show_animation=False, graphics=graphics)

    print("Simulation complete.")

if __name__ == "__main__":
    main()
