"""
Easy evaluation module for ClaimReview2024+ Easy dataset using EasyDynamicSummary procedure.
This module provides specialized evaluation functions for simple difficulty claims.

Key features:
- Uses ClaimReview2024_Easy benchmark with only easy difficulty claims
- Employs EasyDynamicSummary procedure with reduced max_iterations (default: 2)
- Optimized for faster evaluation on simple claims
- Provides convenient wrapper functions for quick testing

Example usage:
    # Quick evaluation with 10 samples
    quick_easy_evaluate(llm="gpt_4o_mini", n_samples=10)
    
    # Full evaluation with custom parameters
    easy_evaluate(
        llm="gpt_4o", 
        n_samples=50, 
        max_iterations=2,
        experiment_name="easy_test"
    )
    
    # Batch evaluation across multiple models
    batch_easy_evaluate(llms=["gpt_4o_mini", "gpt_4o"], n_samples=20)
"""

import csv
import inspect
import json
import re
import time
import traceback
from multiprocessing import Process
from pathlib import Path
from queue import Empty
from typing import Sequence, Optional

import nltk
import numpy as np
import pandas as pd
import torch
import yaml
from nltk.tokenize.treebank import TreebankWordDetokenizer
from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from defame.common import Label, logger, Action
from defame.common.modeling import model_specifier_to_shorthand, AVAILABLE_MODELS, make_model
from defame.eval import load_benchmark
from defame.eval.benchmark import Benchmark
from defame.eval.claimreview2024.benchmark import ClaimReview2024_Easy
from defame.evidence_retrieval.tools import initialize_tools
from defame.fact_checker import FactChecker
from defame.helpers.parallelization.pool import Pool
from defame.helpers.parallelization.task import Task
from defame.utils.console import bold, sec2hhmmss, sec2mmss, num2text
from defame.utils.plot import plot_confusion_matrix
from defame.utils.utils import unroll_dict
from defame.eval.evaluate import (
    validate_config, 
    aggregate_stats, 
    finalize_evaluation, 
    compute_metrics, 
    save_stats
)


def easy_evaluate(
        llm: str = "gpt_4o_mini",
        tools_config: Optional[dict[str, dict]] = None,
        experiment_name: Optional[str] = None,
        fact_checker_kwargs: Optional[dict] = None,
        llm_kwargs: Optional[dict] = None,
        benchmark_kwargs: Optional[dict] = None,
        allowed_actions: Optional[list[str]] = None,
        n_samples: Optional[int] = None,
        sample_ids: Optional[list[int | str]] = None,
        random_sampling: bool = False,
        print_log_level: str = "log",
        continue_experiment_dir: Optional[str] = None,
        n_workers: Optional[int] = None,
        max_iterations: int = 2,
):
    """
    Evaluate ClaimReview2024+ Easy dataset using EasyDynamicSummary procedure.
    
    Args:
        llm: Language model to use (default: "gpt_4o_mini")
        tools_config: Configuration for evidence retrieval tools
        experiment_name: Name of the experiment
        fact_checker_kwargs: Additional arguments for FactChecker
        llm_kwargs: Additional arguments for LLM
        benchmark_kwargs: Additional arguments for benchmark
        allowed_actions: List of allowed action names
        n_samples: Number of samples to evaluate
        sample_ids: Specific sample IDs to evaluate
        random_sampling: Whether to use random sampling
        print_log_level: Logging level
        continue_experiment_dir: Directory to continue previous experiment
        n_workers: Number of parallel workers
        max_iterations: Maximum iterations for easy procedure (default: 2)
    """
    assert not n_samples or not sample_ids

    # Set default configurations for easy evaluation
    if tools_config is None:
        tools_config = dict(searcher={})
    
    if llm_kwargs is None:
        llm_kwargs = dict()

    if fact_checker_kwargs is None:
        fact_checker_kwargs = dict()
    
    # Force easy procedure variant and set max_iterations
    fact_checker_kwargs["procedure_variant"] = "easy"
    fact_checker_kwargs["max_iterations"] = max_iterations

    if benchmark_kwargs is None:
        benchmark_kwargs = dict()

    logger.set_log_level(print_log_level)

    # Load ClaimReview2024_Easy benchmark
    benchmark = ClaimReview2024_Easy(**benchmark_kwargs)

    is_resumed = continue_experiment_dir is not None
    status_verb = "Resuming" if is_resumed else "Starting"
    exp_name_str = f" '{bold(experiment_name)}'" if experiment_name else ""
    logger.info(f"{status_verb} easy evaluation{exp_name_str} on {benchmark.name}.")

    llm = model_specifier_to_shorthand(llm) if llm not in AVAILABLE_MODELS["Shorthand"].values else llm

    procedure_variant = fact_checker_kwargs.get("procedure_variant", "easy")

    logger.set_experiment_dir(path=continue_experiment_dir,
                              benchmark_name=benchmark.shorthand,
                              procedure_name=procedure_variant,
                              model_name=llm,
                              experiment_name=experiment_name)
    logger.log("Saving all outputs to:", logger.target_dir.as_posix())

    n_devices = torch.cuda.device_count()
    if n_workers is None:
        match llm:
            case "llama3_8b":
                n_workers = 8
            case "llama3_70b":
                n_workers = 3  # only 3 copies fit on 8 A100 GPUs
            case _:
                n_workers = max(1, n_devices * 2)  # 2 workers per GPU, minimum 1

    # Save hyperparams based on the signature of easy_evaluate()
    if not is_resumed:
        signature = inspect.signature(easy_evaluate)
        logger.save_config(signature, locals())

    if allowed_actions is None:
        allowed_actions = [a.name for a in benchmark.available_actions]
    else:
        # Filter actions to only include those available in the benchmark
        available_action_names = [a.name for a in benchmark.available_actions]
        allowed_actions = [a for a in allowed_actions if a in available_action_names]

    # Sanity check - convert action names back to Action objects for validation
    action_objects = [a for a in benchmark.available_actions if a.name in allowed_actions]
    p = Process(target=validate_config, args=(tools_config, action_objects))
    p.start()
    p.join()

    if random_sampling:
        benchmark.shuffle()

    if n_samples:
        assert 0 < n_samples <= len(benchmark), f"{n_samples} specified but only {len(benchmark)} samples available."
        samples = benchmark[:n_samples]
    elif sample_ids:
        samples = [benchmark.get_by_id(str(i)) for i in sample_ids]
    else:
        samples = benchmark

    # Exclude already existing samples (relevant if evaluation is resumed)
    if is_resumed:
        samples_to_evaluate = []
        # Retrieve the IDs of already checked claims
        predictions_path = continue_experiment_dir + "/predictions.csv"
        df = pd.read_csv(predictions_path)
        checked_claim_ids = df["sample_index"].to_numpy()

        # Only keep samples that haven't been checked yet
        for sample in samples:
            # sample["id"] should be convertable into a number type for indexing
            if int(sample["id"]) not in checked_claim_ids:
                samples_to_evaluate.append(sample)

        stats_file_path = logger.target_dir / 'results.json'
        if stats_file_path.exists():
            with open(stats_file_path, "r") as f:
                stats = json.load(f)
        else:
            stats = dict()

    else:
        samples_to_evaluate = samples
        stats = dict()

    # Update number of to-be-checked samples
    n_samples = len(samples_to_evaluate)

    if n_samples == 0:
        raise RuntimeError("Nothing to evaluate.")

    n_workers = min(n_workers, n_samples)

    start_time = time.time()

    print(f"Evaluating {n_samples} easy samples using {n_workers} workers...")

    pool = Pool(n_workers=n_workers,
                llm=llm,
                llm_kwargs=llm_kwargs,
                tools_config=tools_config,
                available_actions=action_objects,  # Use Action objects, not strings
                class_definitions=benchmark.class_definitions,
                extra_prepare_rules=benchmark.extra_prepare_rules,
                extra_plan_rules=benchmark.extra_plan_rules,
                extra_judge_rules=benchmark.extra_judge_rules,
                print_log_level=print_log_level,
                target_dir=logger.target_dir,
                **fact_checker_kwargs)

    # Turn each sample into a task and add it to the pool's task queue
    for instance in samples_to_evaluate:
        task = Task(instance["input"], id=instance["id"])
        pool.add_task(task)

    progress = tqdm(range(n_samples), smoothing=0.02)

    pool.wait_until_ready()

    try:
        while progress.n + pool.n_failed_tasks < n_samples:
            try:
                output = pool.get_result(timeout=60)
                benchmark.process_output(output)
                progress.update(1)

            except Empty as e:
                if not pool.is_running():
                    logger.warning("Worker pool stopped running early. Terminating evaluation.")
                    break

    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main process:")
        logger.critical(traceback.format_exc())

    end_time = time.time()
    duration = end_time - start_time

    stats.update({
        "Number of workers": n_workers,
        "Total run duration": duration + stats.get("Total run duration", 0),
        "Max iterations": max_iterations,
        "Procedure": "EasyDynamicSummary"
    })

    finalize_evaluation(logger.target_dir, benchmark, stats)


def quick_easy_evaluate(
        llm: str = "gpt_4o_mini",
        n_samples: int = 10,
        experiment_name: Optional[str] = None,
        max_iterations: int = 2,
        **kwargs
):
    """
    Quick evaluation function for easy testing with minimal configuration.
    
    Args:
        llm: Language model to use
        n_samples: Number of samples to evaluate (default: 10)
        experiment_name: Name of the experiment
        max_iterations: Maximum iterations for easy procedure
        **kwargs: Additional arguments passed to easy_evaluate
    """
    if experiment_name is None:
        experiment_name = f"quick_easy_{n_samples}_samples"
    
    return easy_evaluate(
        llm=llm,
        n_samples=n_samples,
        experiment_name=experiment_name,
        max_iterations=max_iterations,
        **kwargs
    )


def batch_easy_evaluate(
        llms: Optional[list[str]] = None,
        n_samples: Optional[int] = None,
        experiment_prefix: str = "easy_batch",
        max_iterations: int = 2,
        **kwargs
):
    """
    Batch evaluation across multiple models for easy difficulty claims.
    
    Args:
        llms: List of language models to evaluate
        n_samples: Number of samples to evaluate for each model
        experiment_prefix: Prefix for experiment names
        max_iterations: Maximum iterations for easy procedure
        **kwargs: Additional arguments passed to easy_evaluate
    """
    if llms is None:
        llms = ["gpt_4o_mini", "gpt_4o"]
    
    results = {}
    
    for llm in llms:
        experiment_name = f"{experiment_prefix}_{llm}"
        logger.info(f"Starting batch evaluation for {llm}")
        
        try:
            easy_evaluate(
                llm=llm,
                n_samples=n_samples,
                experiment_name=experiment_name,
                max_iterations=max_iterations,
                **kwargs
            )
            results[llm] = "Success"
        except Exception as e:
            logger.error(f"Error evaluating {llm}: {e}")
            results[llm] = f"Error: {e}"
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Running quick easy evaluation...")
    quick_easy_evaluate(
        llm="gpt_4o_mini",
        n_samples=5,
        experiment_name="test_easy_eval",
        max_iterations=2
    )
