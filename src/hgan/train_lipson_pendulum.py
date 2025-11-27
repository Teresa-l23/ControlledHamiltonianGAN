#!/usr/bin/env python
"""
Training script for Lipson Pendulum dataset.
Trains an unconditional CHGAN model on real pendulum trajectory data.
"""
import os
import sys
import argparse
import logging
from hgan.configuration import load_config
from hgan.experiment import Experiment
from hgan.utils import setup_reproducibility

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Train CHGAN on Lipson Pendulum dataset")
    parser.add_argument(
        "--config-path",
        type=str,
        default="hgan/lipson_pendulum_real.ini",
        help="Path to configuration.ini specifying experiment parameters",
    )
    parser.add_argument(
        "--lipson-experiment",
        type=str,
        default="pend-real",
        choices=["pend-real", "pend-sim"],
        help="Which Lipson dataset to use (real or simulated pendulum)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    
    return parser


def main(*args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    args = get_parser().parse_args(args)
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Override config with command line arguments
    if args.output_dir:
        config.paths.output = args.output_dir
    
    # Set Lipson experiment type
    config.experiment.lipson_experiment = args.lipson_experiment
    
    logger.info(f"Configuration loaded from: {args.config_path}")
    logger.info(f"Output directory: {config.paths.output}")
    logger.info(f"Lipson experiment: {args.lipson_experiment}")
    logger.info(f"Number of epochs: {config.experiment.n_epoch}")
    logger.info(f"Batch size: {config.experiment.batch_size}")
    logger.info(f"Learning rate: {config.experiment.learning_rate}")
    logger.info(f"Architecture: {config.experiment.architecture}")
    
    # Setup reproducibility
    setup_reproducibility(seed=config.experiment.seed)
    
    # Create output directory
    os.makedirs(config.paths.output, exist_ok=True)
    
    # Initialize experiment
    logger.info("Initializing experiment...")
    experiment = Experiment(config)
    
    # Check dataset
    logger.info(f"Dataset size: {len(experiment.dataset)}")
    logger.info(f"Number of batches per epoch: {len(experiment.dataloader)}")
    
    # Print model info
    total_params_g = sum(p.numel() for p in experiment.Gi.parameters())
    total_params_d = sum(p.numel() for p in experiment.Dv.parameters())
    total_params_rnn = sum(p.numel() for p in experiment.rnn.parameters())
    
    logger.info(f"Generator parameters: {total_params_g:,}")
    logger.info(f"Discriminator parameters: {total_params_d:,}")
    logger.info(f"RNN parameters: {total_params_rnn:,}")
    logger.info(f"Total parameters: {total_params_g + total_params_d + total_params_rnn:,}")
    
    # Start training
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    try:
        experiment.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving current state...")
        # Save current state before exiting
        current_epoch = len(experiment.saved_epochs())
        if current_epoch > 0:
            experiment.save_epoch(current_epoch)
        logger.info("State saved. Exiting.")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Models saved to: {config.paths.output}")


if __name__ == "__main__":
    main(*sys.argv[1:])
