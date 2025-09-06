"""Use this script to resume an incomplete evaluation run."""

# experiment_dir = "out/verite/summary/dynamic/gpt_4o/2025-08-24_18-29 verite"
experiment_dir = "out/verite/summary/dynamic/gpt_4o/2025-08-24_18-29 verite"
if __name__ == '__main__':  # evaluation uses multiprocessing
    from multiprocessing import set_start_method
    set_start_method("spawn")
    from defame.eval.continute import continue_evaluation
    continue_evaluation(experiment_dir=experiment_dir)
