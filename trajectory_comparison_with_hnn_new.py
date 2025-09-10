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

# 添加项目路径
PROJECT_PATH = '/home/jiayinliu/Desktop/ControlledHamiltonianGAN_0'
sys.path.append(PROJECT_PATH)
sys.path.append(os.path.join(PROJECT_PATH, 'src'))

# 添加hamiltonian-nn路径
HAMILTONIAN_NN_PATH = os.path.join(PROJECT_PATH, 'hamiltonian-nn')
sys.path.append(HAMILTONIAN_NN_PATH)

# 导入HGAN模块
try:
    from hgan.hgn.environments.environment_factory import EnvFactory
    from hgan.hgn_datasets import constant_physics_hgn
    HGAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠ HGAN modules not available: {e}")
    HGAN_AVAILABLE = False

# 导入HNN模块
try:
    from nn_models import MLP
    from hnn import HNN
    from utils import integrate_model
    HNN_AVAILABLE = True
except ImportError as e:
    print(f"⚠ HNN modules not available: {e}")
    HNN_AVAILABLE = False


def get_custom_hnn_model_path(system_name, mask=None):
    """根据系统名称返回自定义训练的HNN模型路径"""
    # 系统名称到HGN系统名称的映射
    system_to_hgn = {
        'Spring': 'mass_spring',
        'Pendulum': 'pendulum', 
        'ChaoticPendulum': 'double_pendulum',
        'NObjectGravity': 'two_body'  # 默认为two_body
    }
    
    if system_name == 'NObjectGravity' and mask is not None:
        num_objects = np.sum(mask > 0)
        if num_objects == 3:
            system_to_hgn['NObjectGravity'] = 'three_body'
    
    hgn_system = system_to_hgn.get(system_name, 'two_body')
    model_filename = f'custom_hnn_{hgn_system}_model.pth'
    return os.path.join(PROJECT_PATH, model_filename)

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
            'ChaoticPendulum': 'double_pendulum'
        }
        system_folder = system_mapping.get(system_name, system_name.lower())
    
    results_dir = os.path.join(PROJECT_PATH, 'results', system_folder)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def load_custom_hnn_model(model_path, input_dim=8, hidden_dim=200):
    """Load custom trained HNN model from .pth file"""
    try:
        # Import required modules
        from nn_models import MLP
        from hnn import HNN
        
        # Create model architecture (matching retrain script)
        nn_model = MLP(input_dim, hidden_dim, 2, 'tanh')
        model = HNN(input_dim, differentiable_model=nn_model,
                   field_type='solenoidal', baseline=False)
        
        # Load state dict from .pth file
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model
        
    except Exception as e:
        print(f"Failed to load custom HNN model: {e}")
        return None


def generate_hnn_trajectory(hnn_model, q_initial, p_initial, n_steps=30, dt=0.05, model_path=None):
    """使用HNN模型生成轨迹"""
    if hnn_model is None:
        return None
        
    total_time = dt * (n_steps - 1)
    t_span = (0, total_time)
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
            
    if q_initial.ndim == 0:
        q_initial = np.array([q_initial])
    if p_initial.ndim == 0:
        p_initial = np.array([p_initial])
    hnn_initial_condition = np.concatenate([q_initial, p_initial])
        
    sol = integrate_model(hnn_model, t_span, hnn_initial_condition, 
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)
    trajectory = sol.y.T
    return trajectory

def main():
    if not HGAN_AVAILABLE:
        print("Error: HGAN modules not available!")
        return
        
    # Load data
    data_file = os.path.join(PROJECT_PATH, 'data/2P/comp_047900_data.npz')
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    data = np.load(data_file)
    fake_traj = data['fake']
    real_traj = data['real'] 
    mask = data['mask']
    system_name = data['system_name'].item()
    
    print(f"Processing {system_name} system with {np.sum(mask > 0)} particles...")
    
    # Determine dataset system name based on system_name and mask
    if system_name == 'NObjectGravity':
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
            'ChaoticPendulum': 'double_pendulum'
        }
        dataset_system_name = system_mapping.get(system_name, 'two_body')
    
    # Get system args and create environment
    system_args = constant_physics_hgn[dataset_system_name]
    system_args = {
        k: ([item() if callable(item) else item for item in v] if isinstance(v, list) else 
            (v() if callable(v) else v))
        for k, v in system_args.items()
    }
    
    system = EnvFactory.get_environment(system_name, **system_args)
    
    # Extract q and p from fake trajectory
    q_fake = system.extract_q(fake_traj, mask, frame_idx=0)
    q_fake_next = system.extract_q(fake_traj, mask, frame_idx=1)
    
    # Calculate momentum
    delta = 0.05
    mass = np.atleast_1d(system_args["mass"])
    p_fake = mass[:, None] * (q_fake_next - q_fake) / delta

    # Load custom HNN model
    hnn_model_path = get_custom_hnn_model_path(system_name, mask)
    input_dim = q_fake.size + p_fake.size
    print(f"Loading HNN model: {os.path.basename(hnn_model_path)}")
    hnn_model = load_custom_hnn_model(hnn_model_path, input_dim=input_dim)
    
    if hnn_model is None:
        print("Failed to load HNN model")
        return
    
    # Generate HNN trajectory
    hnn_traj = generate_hnn_trajectory(
        hnn_model, q_fake.flatten(), p_fake.flatten(), 
        n_steps=fake_traj.shape[0], dt=0.05
    )
    
    if hnn_traj is None:
        print("Failed to generate HNN trajectory")
        return
    
    system._rollout = hnn_traj.transpose()
    hnn_coords_full = system._convert_to_t2n_format()

    # Get all valid particles for comparison
    valid_idx = np.where(mask > 0)[0]
    if len(valid_idx) == 0:
        print("No valid particles found!")
        return
    
    n_valid_particles = len(valid_idx)
    
    # Create results directory
    results_dir = create_results_directory(system_name, mask)
    
    # Calculate MSE for each particle and overall mean
    particle_mses_hgan = []
    particle_mses_hnn = []
    particle_pos_mses_hgan = []
    particle_pos_mses_hnn = []
    particle_vel_mses_hgan = []
    particle_vel_mses_hnn = []
    
    # Create visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, n_valid_particles, figsize=(5*n_valid_particles, 10))
    if n_valid_particles == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, particle_idx in enumerate(valid_idx):
        # Extract coordinates for this particle
        real_coords = real_traj[:, :, particle_idx]  # (T, 2)
        fake_coords = fake_traj[:, :, particle_idx]  # (T, 2)
        
        # Extract HNN coordinates for this particle
        if hnn_coords_full.ndim == 3:
            hnn_coords = hnn_coords_full[:, :, particle_idx]  # (T, 2)
        else:
            # If HNN only has one particle, use it for all
            hnn_coords = hnn_coords_full[:, :]  # (T, 2)
        
        # Calculate MSE for this particle
        hgan_mse = np.mean((fake_coords - real_coords) ** 2)
        hgan_pos_mse = np.mean((fake_coords[:, 0] - real_coords[:, 0]) ** 2)
        hgan_vel_mse = np.mean((fake_coords[:, 1] - real_coords[:, 1]) ** 2)
        
        hnn_mse = np.mean((hnn_coords - real_coords) ** 2)
        hnn_pos_mse = np.mean((hnn_coords[:, 0] - real_coords[:, 0]) ** 2)
        hnn_vel_mse = np.mean((hnn_coords[:, 1] - real_coords[:, 1]) ** 2)
        
        particle_mses_hgan.append(hgan_mse)
        particle_mses_hnn.append(hnn_mse)
        particle_pos_mses_hgan.append(hgan_pos_mse)
        particle_pos_mses_hnn.append(hnn_pos_mse)
        particle_vel_mses_hgan.append(hgan_vel_mse) 
        particle_vel_mses_hnn.append(hnn_vel_mse)
        
        # Plot position trajectories
        ax_pos = axes[0, i]
        ax_pos.plot(real_coords[:, 0], real_coords[:, 1], 'k-', linewidth=2, label='Real', alpha=0.8)
        ax_pos.plot(fake_coords[:, 0], fake_coords[:, 1], 'b--', linewidth=2, label='HGAN', alpha=0.7)
        ax_pos.plot(hnn_coords[:, 0], hnn_coords[:, 1], 'r:', linewidth=2, label='HNN', alpha=0.7)
        ax_pos.set_title(f'Particle {particle_idx} - Position Trajectory')
        ax_pos.set_xlabel('X Position')
        ax_pos.set_ylabel('Y Position')
        ax_pos.legend()
        ax_pos.grid(True, alpha=0.3)
        ax_pos.axis('equal')
        
        # Plot time series
        ax_time = axes[1, i]
        time_steps = np.arange(len(real_coords))
        ax_time.plot(time_steps, real_coords[:, 0], 'k-', linewidth=2, label='Real X', alpha=0.8)
        ax_time.plot(time_steps, fake_coords[:, 0], 'b--', linewidth=2, label='HGAN X', alpha=0.7)
        ax_time.plot(time_steps, hnn_coords[:, 0], 'r:', linewidth=2, label='HNN X', alpha=0.7)
        ax_time.plot(time_steps, real_coords[:, 1], 'k-', linewidth=1, label='Real Y', alpha=0.6)
        ax_time.plot(time_steps, fake_coords[:, 1], 'b--', linewidth=1, label='HGAN Y', alpha=0.5)
        ax_time.plot(time_steps, hnn_coords[:, 1], 'r:', linewidth=1, label='HNN Y', alpha=0.5)
        ax_time.set_title(f'Particle {particle_idx} - Time Series')
        ax_time.set_xlabel('Time Step')
        ax_time.set_ylabel('Coordinate Value')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to results directory
    plot_filename = os.path.join(results_dir, f'trajectory_comparison_{n_valid_particles}particles.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close plot instead of showing
    
    # Calculate mean MSE across all particles
    mean_hgan_mse = np.mean(particle_mses_hgan)
    mean_hnn_mse = np.mean(particle_mses_hnn)
    mean_hgan_pos_mse = np.mean(particle_pos_mses_hgan)
    mean_hnn_pos_mse = np.mean(particle_pos_mses_hnn)
    mean_hgan_vel_mse = np.mean(particle_vel_mses_hgan)
    mean_hnn_vel_mse = np.mean(particle_vel_mses_hnn)
    
    # Print summary results
    print(f"Results:")
    print(f"  HGAN Total MSE: {mean_hgan_mse:.6f}")
    print(f"  HNN Total MSE:  {mean_hnn_mse:.6f}")
    print(f"  MSE Ratio (HGAN/HNN): {mean_hgan_mse/mean_hnn_mse:.3f}")
    print(f"  {'✓ HGAN better' if mean_hgan_mse < mean_hnn_mse else '✗ HNN better'}")
    
    # Save MSE results to JSON file in results directory
    results = {
        'system_name': system_name,
        'n_valid_particles': n_valid_particles,
        'trajectory_length': fake_traj.shape[0],
        'hnn_model_path': os.path.basename(hnn_model_path),
        'particle_mses': {
            'hgan': particle_mses_hgan,
            'hnn': particle_mses_hnn,
            'hgan_pos': particle_pos_mses_hgan,
            'hnn_pos': particle_pos_mses_hnn,
            'hgan_vel': particle_vel_mses_hgan,
            'hnn_vel': particle_vel_mses_hnn
        },
        'mean_mses': {
            'hgan_total': mean_hgan_mse,
            'hnn_total': mean_hnn_mse,
            'hgan_pos': mean_hgan_pos_mse,
            'hnn_pos': mean_hnn_pos_mse,
            'hgan_vel': mean_hgan_vel_mse,
            'hnn_vel': mean_hnn_vel_mse
        },
        'improvement_ratios': {
            'total': mean_hgan_mse/mean_hnn_mse,
            'position': mean_hgan_pos_mse/mean_hnn_pos_mse,
            'velocity': mean_hgan_vel_mse/mean_hnn_vel_mse
        }
    }
    
    results_filename = os.path.join(results_dir, f'mse_comparison_{n_valid_particles}particles.json')
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_filename}")

if __name__ == "__main__":
    main()
