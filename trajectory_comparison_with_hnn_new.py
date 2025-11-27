#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse
import pickle
import json
import glob
from collections import defaultdict

# 添加项目路径
PROJECT_PATH = '/home/jiayinliu/Desktop/Experiment/LowDimConditionalHGan/'
sys.path.append(os.path.join(PROJECT_PATH, 'ControlledHamiltonianGAN_11/src'))
sys.path.append(os.path.join(PROJECT_PATH, 'ControlledHamiltonianGAN_11'))

# 导入HGAN模块
try:
    from hgan.hgn.environments.environment_factory import EnvFactory
    from hgan.hgn_datasets import constant_physics_hgn
    HGAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠ HGAN modules not available: {e}")
    HGAN_AVAILABLE = False

def create_results_directory(system_name, mask=None):
    """创建结果目录结构"""
    # 确定系统名称
    if system_name == 'NObjectGravity' and mask is not None:
        num_objects = np.sum(mask > 0)
        system_folder = f"{num_objects}_body"
    else:
        system_mapping = {
            'Spring': 'mass_spring',
            'Pendulum': 'pendulum',
            'ChaoticPendulum': 'double_pendulum',
            "PitchforkBifurcation": "pitchfork_bifurcation",
            "SaddleNodeBifurcation": "saddle_node_bifurcation",
            'pend-real': 'lipson_pendulum_real',
            'pend-sim': 'lipson_pendulum_sim',
        }
        system_folder = system_mapping.get(system_name, system_name.lower())
    
    results_dir = os.path.join(PROJECT_PATH, 'results', system_folder)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def process_single_file_lipson(data_file, rollout_lengths=None):
    """处理Lipson数据集文件 - 直接比较生成轨迹与真实轨迹"""
    if rollout_lengths is None:
        rollout_lengths = [30]  # Default full trajectory
    
    data = np.load(data_file)
    fake_traj = data['fake']  # Generated trajectory
    real_traj = data.get('real', fake_traj)  # Real trajectory (if available)
    mask = data['mask']
    system_name = data['system_name'].item() if isinstance(data['system_name'], np.ndarray) else data['system_name']
    
    # Get valid particles
    valid_idx = np.where(mask > 0)[0]
    if len(valid_idx) == 0:
        return None
    
    # Store results for each rollout length
    results_by_length = {}
    
    for n_steps in rollout_lengths:
        if n_steps > min(fake_traj.shape[0], real_traj.shape[0]):
            continue  # Skip if requested length exceeds available data
        
        # Truncate trajectories to current rollout length
        fake_traj_truncated = fake_traj[:n_steps]
        real_traj_truncated = real_traj[:n_steps]
        
        # Calculate MSE for each particle
        particle_mses_hgan = []
        
        for i, particle_idx in enumerate(valid_idx):
            # Extract coordinates for this particle
            real_coords = real_traj_truncated[:, :, particle_idx]  # (T, 2)
            fake_coords = fake_traj_truncated[:, :, particle_idx]  # (T, 2)
            
            # Calculate MSE for this particle
            hgan_mse = np.mean((fake_coords - real_coords) ** 2)
            
            particle_mses_hgan.append(hgan_mse)
        
        # Store results for this rollout length
        results_by_length[n_steps] = {
            'mean_hgan_mse': np.mean(particle_mses_hgan),
            'particle_mses_hgan': particle_mses_hgan,
            'trajectories': {
                'real': real_traj_truncated,
                'fake': fake_traj_truncated
            }
        }
    
    # Return results for this file
    return {
        'filename': os.path.basename(data_file),
        'system_name': system_name,
        'n_particles': len(valid_idx),
        'results_by_length': results_by_length,
        'mask': mask,
        'valid_idx': valid_idx
    }


def process_single_file(data_file, system, system_args, rollout_lengths=None):
    """处理单个数据文件，支持多个rollout长度"""
    if rollout_lengths is None:
        rollout_lengths = [30]  # Default full trajectory
    
    data = np.load(data_file)
    fake_traj = data['fake']
    mask = data['mask']
    system_name = data['system_name'].item() if isinstance(data['system_name'], np.ndarray) else data['system_name']
    
    # Extract q and p from fake trajectory
    q_fake = system.extract_q(fake_traj, mask, frame_idx=0)
    q_fake_next = system.extract_q(fake_traj, mask, frame_idx=1)
    
    # Calculate momentum
    delta = 0.05
    mass = np.atleast_1d(system_args["mass"])
    p_fake = mass[:, None] * (q_fake_next - q_fake) / delta
    
    # Get all valid particles for comparison
    valid_idx = np.where(mask > 0)[0]
    if len(valid_idx) == 0:
        return None
    
    # Store results for each rollout length
    results_by_length = {}
    
    for n_steps in rollout_lengths:
        if n_steps > fake_traj.shape[0]:
            continue  # Skip if requested length exceeds available data
        
        # Generate real trajectory with specific length
        if system_name == 'NObjectGravity':
            real_traj = system.calculate_fixed_rollout(q_fake, p_fake, n_steps, delta)
        else:
            real_traj = system.calculate_fixed_rollout(q_fake.flatten(), p_fake.flatten(), n_steps, delta)
        
        # Truncate CHGAN trajectory to current rollout length
        fake_traj_truncated = fake_traj[:n_steps]
        
        # Calculate MSE for each particle
        particle_mses_hgan = []
        
        for i, particle_idx in enumerate(valid_idx):
            # Extract coordinates for this particle
            real_coords = real_traj[:, :, particle_idx]  # (T, 2)
            fake_coords = fake_traj_truncated[:, :, particle_idx]  # (T, 2)
            
            # Calculate MSE for this particle
            hgan_mse = np.mean((fake_coords - real_coords) ** 2)
            
            particle_mses_hgan.append(hgan_mse)
        
        # Store results for this rollout length
        results_by_length[n_steps] = {
            'mean_hgan_mse': np.mean(particle_mses_hgan),
            'particle_mses_hgan': particle_mses_hgan,
            'trajectories': {
                'real': real_traj,
                'fake': fake_traj_truncated
            }
        }
    
    # Return results for this file
    return {
        'filename': os.path.basename(data_file),
        'system_name': system_name,
        'n_particles': len(valid_idx),
        'results_by_length': results_by_length,
        'mask': mask,
        'valid_idx': valid_idx
    }


def plot_rollout_length_comparison(results, results_dir, system_name):
    """绘制不同rollout长度的MSE对比"""
    rollout_lengths = sorted(list(results[0]['results_by_length'].keys()))
    
    # Aggregate MSEs across all files for each rollout length
    hgan_mses_by_length = {length: [] for length in rollout_lengths}
    # hnn_mses_by_length = {length: [] for length in rollout_lengths}
    
    for result in results:
        for length, data in result['results_by_length'].items():
            hgan_mses_by_length[length].append(data['mean_hgan_mse'])
            # hnn_mses_by_length[length].append(data['mean_hnn_mse'])
    
    # Calculate mean and std for each length
    hgan_means = [np.mean(hgan_mses_by_length[l]) for l in rollout_lengths]
    hgan_stds = [np.std(hgan_mses_by_length[l]) for l in rollout_lengths]
    # hnn_means = [np.mean(hnn_mses_by_length[l]) for l in rollout_lengths]
    # hnn_stds = [np.std(hnn_mses_by_length[l]) for l in rollout_lengths]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MSE vs Rollout Length
    ax1.errorbar(rollout_lengths, hgan_means, yerr=hgan_stds, 
                 marker='o', linewidth=2, capsize=5, label='CHGAN', color='#3498db')
    ax1.set_xlabel('Rollout Length (timesteps)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean MSE', fontsize=12, fontweight='bold')
    ax1.set_title(f'{system_name} - MSE vs Rollout Length', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    
    plt.tight_layout()
    plot_filename = os.path.join(results_dir, f'rollout_length_comparison_{system_name}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename


def plot_best_trajectory(best_result, results_dir, rollout_length):
    """绘制指定rollout长度的最佳结果轨迹对比图"""
    length_data = best_result['results_by_length'][rollout_length]
    real_traj = length_data['trajectories']['real']
    fake_traj = length_data['trajectories']['fake']
    valid_idx = best_result['valid_idx']
    n_valid_particles = len(valid_idx)
    
    # Create visualization
    fig, axes = plt.subplots(2, n_valid_particles, figsize=(5*n_valid_particles, 10))
    if n_valid_particles == 1:
        axes = axes.reshape(2, 1)
    
    for i, particle_idx in enumerate(valid_idx):
        # Extract coordinates for this particle
        real_coords = real_traj[:, :, particle_idx]  # (T, 2)
        fake_coords = fake_traj[:, :, particle_idx]  # (T, 2)
        

        # Plot position trajectories
        ax_pos = axes[0, i]
        ax_pos.plot(real_coords[:, 0], real_coords[:, 1], 'k-', linewidth=2, label='Real', alpha=0.8)
        ax_pos.plot(fake_coords[:, 0], fake_coords[:, 1], 'b--', linewidth=2, label='CHGAN', alpha=0.7)
        ax_pos.set_title(f'Particle {particle_idx} - Position (Length {rollout_length})\n{best_result["filename"]}')
        ax_pos.set_xlabel('X Position')
        ax_pos.set_ylabel('Y Position')
        ax_pos.legend()
        ax_pos.grid(True, alpha=0.3)
        ax_pos.axis('equal')
        
        # Plot time series
        ax_time = axes[1, i]
        time_steps = np.arange(len(real_coords))
        ax_time.plot(time_steps, real_coords[:, 0], 'k-', linewidth=2, label='Real X', alpha=0.8)
        ax_time.plot(time_steps, fake_coords[:, 0], 'b--', linewidth=2, label='CHGAN X', alpha=0.7)
        ax_time.plot(time_steps, real_coords[:, 1], 'k-', linewidth=1, label='Real Y', alpha=0.6)
        ax_time.plot(time_steps, fake_coords[:, 1], 'b--', linewidth=1, label='CHGAN Y', alpha=0.5)
        ax_time.set_title(f'Particle {particle_idx} - Time Series')
        ax_time.set_xlabel('Time Step')
        ax_time.set_ylabel('Coordinate Value')
        ax_time.legend()  
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to results directory
    plot_filename = os.path.join(results_dir, f'best_trajectory_comparison_{best_result["system_name"]}_length{rollout_length}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def main():
    if not HGAN_AVAILABLE:
        print("Error: HGAN modules not available!")
        return
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Compare HGAN and HNN trajectories for a folder of data files')
    parser.add_argument('--data_folder', type=str, default='/home/jiayinliu/Desktop/ControlledHamiltonianGAN_0/data/1P',
                        help='Path to folder containing .npz data files')
    parser.add_argument('--results_folder', type=str, default='/home/jiayinliu/Desktop/ControlledHamiltonianGAN_0/results',
                        help='Path to save results')
    parser.add_argument('--rollout_lengths', type=int, nargs='+', default=[5, 10, 20, 25, 30],
                        help='List of rollout lengths to compare')
    args = parser.parse_args()
    
    data_folder = args.data_folder
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
        return
    
    # Find all .npz files in the folder
    npz_files = glob.glob(os.path.join(data_folder, '*_data.npz'))
    if not npz_files:
        print(f"No *_data.npz files found in {data_folder}")
        return
    
    print(f"Found {len(npz_files)} data files to process")
    
    # Load first file to determine system type and setup
    first_data = np.load(npz_files[0])
    system_name = first_data['system_name'].item() if isinstance(first_data['system_name'], np.ndarray) else first_data['system_name']
    mask = first_data['mask']
    
    # Check if this is a Lipson dataset (real pendulum data)
    is_lipson_dataset = system_name in ['pend-real', 'pend-sim'] or 'lipson' in data_folder.lower()
    
    if is_lipson_dataset:
        # For Lipson dataset, we don't need physics simulation
        print(f"Detected Lipson dataset: {system_name}")
        dataset_system_name = system_name
        
        # Process all files for Lipson dataset
        results = []
        rollout_lengths = args.rollout_lengths
        
        print(f"\nProcessing {len(npz_files)} Lipson dataset files with rollout lengths: {rollout_lengths}")
        for i, data_file in enumerate(npz_files):
            print(f"Processing {i+1}/{len(npz_files)}: {os.path.basename(data_file)}")
            
            result = process_single_file_lipson(data_file, rollout_lengths=rollout_lengths)
            if result is not None and result['results_by_length']:
                results.append(result)
        
        if not results:
            print("No files were successfully processed!")
            return
        
        print(f"\nSuccessfully processed {len(results)} files")
        
        # Continue with result analysis and plotting (same as before)
        # Skip to result analysis section
    elif system_name == 'NObjectGravity':
        num_objects = np.sum(mask > 0)
        if num_objects == 2:
            dataset_system_name = 'two_body'
        elif num_objects == 3:
            dataset_system_name = 'three_body'
        else:
            dataset_system_name = 'two_body'  # default
    else:
        # Map system names to dataset names
        system_mapping = {
            'Pendulum': 'pendulum',
            'Spring': 'mass_spring',
            'ChaoticPendulum': 'double_pendulum',
            "PitchforkBifurcation": "pitchfork_bifurcation",
            "SaddleNodeBifurcation": "saddle_node_bifurcation",
            'pend-real': 'lipson_pendulum_real',
            'pend-sim': 'lipson_pendulum_sim',
        }
        dataset_system_name = system_mapping.get(system_name, 'two_body')
    
    # Print detected system (for non-Lipson datasets)
    if not is_lipson_dataset:
        print(f"Detected system: {system_name} ({dataset_system_name})")
    
    # For non-Lipson datasets, get system args and create environment
    if not is_lipson_dataset:
        print(f"Setting up environment for {system_name} system...")
        system_args = constant_physics_hgn[dataset_system_name]
    system_args = {
        k: ([item() if callable(item) else item for item in v] if isinstance(v, list) else 
            (v() if callable(v) else v))
        for k, v in system_args.items()
    }
    if dataset_system_name == "saddle_node_bifurcation" or dataset_system_name == "pitchfork_bifurcation":
        system_args["lam"] = 1.0
        print(f"Setting 'lam' parameter to 1.0 for {dataset_system_name} system")
    
        system = EnvFactory.get_environment(system_name, **system_args)
        print(f"✓ Environment created successfully")
        
        # Process all files with specified rollout lengths
        results = []
        rollout_lengths = args.rollout_lengths
        
        print(f"\nProcessing {len(npz_files)} files with rollout lengths: {rollout_lengths}")
        for i, data_file in enumerate(npz_files):
            print(f"Processing {i+1}/{len(npz_files)}: {os.path.basename(data_file)}")
            
            result = process_single_file(data_file, system, system_args, rollout_lengths=rollout_lengths)
            if result is not None and result['results_by_length']:
                results.append(result)
        
        if not results:
            print("No files were successfully processed!")
            return
        
        print(f"\nSuccessfully processed {len(results)} files")
    
    # Calculate overall averages for each rollout length
    print(f"\n=== RESULTS BY ROLLOUT LENGTH ===")
    print(f"System: {system_name} ({dataset_system_name})")
    print(f"Files processed: {len(results)}")
    print(f"\n{'Length':<10}{'CHGAN MSE':<15}")
    print("-" * 25)
    
    overall_results = {}
    for length in rollout_lengths:
        hgan_mses = []
        
        for r in results:
            if length in r['results_by_length']:
                hgan_mses.append(r['results_by_length'][length]['mean_hgan_mse'])
        
        if hgan_mses:
            avg_hgan = np.mean(hgan_mses)
            
            overall_results[length] = {
                'avg_hgan_mse': avg_hgan
            }
            
            print(f"{length:<10}{avg_hgan:<15.6f}")
    
    # Find best performing case for the longest rollout
    longest_length = max(rollout_lengths)
    best_result = min(results, 
                     key=lambda x: x['results_by_length'][longest_length]['mean_hgan_mse'] 
                     if longest_length in x['results_by_length'] else float('inf'))
    
    if longest_length in best_result['results_by_length']:
        best_data = best_result['results_by_length'][longest_length]
        print(f"\n=== BEST PERFORMING CASE (Length {longest_length}) ===")
        print(f"File: {best_result['filename']}")
        print(f"System: {best_result['system_name']} ({best_result['n_particles']} particles)")
        print(f"CHGAN MSE: {best_data['mean_hgan_mse']:.6f}")
    
    # Create results directory for the system type
    system_name = best_result['system_name']
    if system_name == 'NObjectGravity':
        n_particles = best_result['n_particles']
        # results_dir = os.path.join(PROJECT_PATH, 'results', f'{n_particles}_body')
    else:
        system_mapping = {
            'Spring': 'mass_spring',
            'Pendulum': 'pendulum',
            'ChaoticPendulum': 'double_pendulum',
            'pend-real': 'lipson_pendulum_real',
            'pend-sim': 'lipson_pendulum_sim',
        }
        system_folder = system_mapping.get(system_name, system_name.lower())
        # results_dir = os.path.join(PROJECT_PATH, 'results', system_folder)
    
    results_dir = args.results_folder
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot rollout length comparison
    comparison_plot = plot_rollout_length_comparison(results, results_dir, dataset_system_name)
    print(f"\nRollout length comparison plot saved to: {comparison_plot}")
    
    # Plot best trajectory for each rollout length
    for length in rollout_lengths:
        if length in best_result['results_by_length']:
            plot_filename = plot_best_trajectory(best_result, results_dir, length)
            print(f"Best trajectory plot (length {length}) saved to: {plot_filename}")
    
    # Save comprehensive results to JSON
    comprehensive_results = {
        'data_folder': data_folder,
        'system_name': system_name,
        'dataset_system_name': dataset_system_name,
        'processed_files': len(results),
        'total_files_found': len(npz_files),
        'rollout_lengths': rollout_lengths,
        'overall_averages_by_length': overall_results,
        'best_case': {
            'filename': best_result['filename'],
            'system_name': best_result['system_name'],
            'n_particles': best_result['n_particles'],
            'results_by_length': {
                str(length): {
                    'chgan_mse': best_result['results_by_length'][length]['mean_hgan_mse']
                }
                for length in rollout_lengths if length in best_result['results_by_length']
            }
        },
        'individual_results': [
            {
                'filename': r['filename'],
                'system_name': r['system_name'], 
                'n_particles': r['n_particles'],
                'results_by_length': {
                    str(length): {
                        'chgan_mse': r['results_by_length'][length]['mean_hgan_mse']
                    }
                    for length in rollout_lengths if length in r['results_by_length']
                }
            }
            for r in results
        ]
    }
    
    results_json_path = os.path.join(results_dir, 'folder_comparison_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"Comprehensive results saved to: {results_json_path}")
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Processed {len(results)}/{len(npz_files)} files successfully")
    print(f"Results saved in: {results_dir}")
    print(f"Best case visualization: {os.path.basename(plot_filename)}")

if __name__ == "__main__":
    main()
