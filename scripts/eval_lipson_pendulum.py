#!/usr/bin/env python
"""
Evaluation script for Lipson Pendulum dataset.
Generates fake trajectories and compares them with real trajectories.
"""
import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch.autograd import Variable
from hgan.configuration import load_config
from hgan.experiment import Experiment
from hgan.utils import setup_reproducibility
from hgan.dataset import LipsonPendulumDataset

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate trained model on Lipson Pendulum dataset")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.ini specifying experiment parameters",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to load (default: latest)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of fake trajectories to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: config output dir)",
    )
    parser.add_argument(
        "--lipson-experiment",
        type=str,
        default="pend-real",
        choices=["pend-real", "pend-sim"],
        help="Which Lipson dataset to use for comparison",
    )
    parser.add_argument(
        "--lipson-data-dir",
        type=str,
        default=None,
        help="Directory containing Lipson dataset (default: hamiltonian-nn/experiment-real)",
    )
    parser.add_argument(
        "--use-test-set",
        action="store_true",
        help="Use test set for evaluation (default: False)",
    )
    
    return parser


def generate_fake_trajectory(experiment, n_frames=None):
    """
    Generate a single fake trajectory using the trained model.
    
    Args:
        experiment: Experiment object with loaded model
        n_frames: Number of frames to generate (default: from config)
    
    Returns:
        fake_trajectory: (T, 2, 1) numpy array
    """
    if n_frames is None:
        n_frames = experiment.config.video.generator_frames
    
    # Create empty label and props for unconditional generation
    label_and_props = torch.tensor([]).unsqueeze(0).to(experiment.device)
    
    # Create mask for single particle
    mask = torch.ones(1, 1, dtype=torch.float32).to(experiment.device)
    
    # Generate fake data
    fake_data = experiment.get_fake_data(
        n_frames=n_frames,
        label_and_props=label_and_props,
        mask=mask
    )
    
    # Extract trajectory (batch_size, T, 2, 1)
    fake_trajectory = fake_data["videos"].detach().cpu().numpy()[0]  # (T, 2, 1)
    
    return fake_trajectory


def evaluate_model(experiment, dataset, n_samples=10, output_dir=None, use_test_set=False):
    """
    Evaluate the trained model by generating fake trajectories and comparing with real ones.
    
    Args:
        experiment: Experiment object with loaded model
        dataset: LipsonPendulumDataset for comparison
        n_samples: Number of fake trajectories to generate
        output_dir: Directory to save results
        use_test_set: Whether to use test set for comparison
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    experiment.eval()
    
    if output_dir is None:
        output_dir = experiment.config.paths.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get full real trajectory for comparison
    real_trajectory, real_mask = dataset.get_full_trajectory(train=not use_test_set)
    real_trajectory_np = real_trajectory.numpy()
    
    logger.info(f"Real trajectory shape: {real_trajectory_np.shape}")
    logger.info(f"Generating {n_samples} fake trajectories...")
    
    results = {
        'best_rmses': [],
        'best_indices': [],
        'all_rmses': [],
    }
    
    for i in range(n_samples):
        logger.info(f"Generating fake trajectory {i+1}/{n_samples}")
        
        # Generate fake trajectory
        fake_trajectory = generate_fake_trajectory(experiment)
        
        logger.info(f"Fake trajectory shape: {fake_trajectory.shape}")
        
        # Find best match in real trajectory
        best_idx, best_rmse, all_rmses = dataset.find_best_match(
            fake_trajectory, real_trajectory_np
        )
        
        results['best_rmses'].append(best_rmse)
        results['best_indices'].append(best_idx)
        results['all_rmses'].append(all_rmses)
        
        logger.info(f"Best match at index {best_idx} with RMSE={best_rmse:.6f}")
        
        # Plot comparison
        save_path = os.path.join(output_dir, f"eval_comparison_{i:03d}.png")
        dataset.plot_trajectory_comparison(
            fake_trajectory,
            real_trajectory_np,
            save_path=save_path,
            title=f"Sample {i+1}/{n_samples}",
            find_best_match=True
        )
        
        # Save trajectory data
        np.savez(
            os.path.join(output_dir, f"eval_trajectory_{i:03d}.npz"),
            fake=fake_trajectory,
            real=real_trajectory_np,
            best_idx=best_idx,
            best_rmse=best_rmse
        )
    
    # Compute summary statistics
    results['mean_rmse'] = np.mean(results['best_rmses'])
    results['std_rmse'] = np.std(results['best_rmses'])
    results['min_rmse'] = np.min(results['best_rmses'])
    results['max_rmse'] = np.max(results['best_rmses'])
    
    # Save summary
    summary_path = os.path.join(output_dir, "eval_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Lipson Pendulum Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Dataset: {'Test' if use_test_set else 'Train'}\n")
        f.write(f"Real trajectory length: {len(real_trajectory_np)}\n")
        f.write(f"Fake trajectory length: {len(fake_trajectory)}\n\n")
        f.write("RMSE Statistics:\n")
        f.write(f"  Mean: {results['mean_rmse']:.6f}\n")
        f.write(f"  Std:  {results['std_rmse']:.6f}\n")
        f.write(f"  Min:  {results['min_rmse']:.6f}\n")
        f.write(f"  Max:  {results['max_rmse']:.6f}\n\n")
        f.write("Individual Results:\n")
        for i, (rmse, idx) in enumerate(zip(results['best_rmses'], results['best_indices'])):
            f.write(f"  Sample {i+1}: RMSE={rmse:.6f}, Best match at t={idx}\n")
    
    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  Mean RMSE: {results['mean_rmse']:.6f} ± {results['std_rmse']:.6f}")
    logger.info(f"  Min RMSE:  {results['min_rmse']:.6f}")
    logger.info(f"  Max RMSE:  {results['max_rmse']:.6f}")
    logger.info(f"\nResults saved to {output_dir}")
    
    return results


def main(*args):
    logging.basicConfig(level=logging.INFO)
    
    args = get_parser().parse_args(args)
    config = load_config(args.config_path)
    
    setup_reproducibility(config.experiment.seed)
    
    # Initialize experiment
    experiment = Experiment(config)
    
    # Load trained model
    if args.output_dir:
        # Load from custom output directory
        experiment.config.paths.output = args.output_dir
    else:
        # Use output directory from config
        args.output_dir = experiment.config.paths.output
    
    epoch = experiment.load_epoch(epoch=args.epoch)
    logger.info(f"Loaded model from epoch {epoch}")
    
    # Initialize Lipson dataset for comparison
    lipson_data_dir = args.lipson_data_dir
    if lipson_data_dir is None:
        # Default to hamiltonian-nn/experiment-real directory
        lipson_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "hamiltonian-nn",
            "experiment-real"
        )
    
    dataset = LipsonPendulumDataset(
        experiment_name=args.lipson_experiment,
        save_dir=lipson_data_dir,
        train=not args.use_test_set,
        test_split=0.8,
        num_frames=config.video.generator_frames,
        delta=None,
    )
    
    logger.info(f"Using Lipson dataset: {args.lipson_experiment}")
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Evaluate
    results = evaluate_model(
        experiment,
        dataset,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        use_test_set=args.use_test_set
    )
    
    return results


if __name__ == "__main__":
    main(*sys.argv[1:])
