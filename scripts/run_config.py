"""Runs the experiment as configured by the specified configuration file."""
import os
if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method("spawn")
    from defame.eval.utils import run_experiment
    run_experiment(config_file_path="config/experiments/claimreview2024.yaml")
