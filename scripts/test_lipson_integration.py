#!/usr/bin/env python
"""
Quick test to verify Lipson Pendulum integration works correctly.
"""
import sys
import os
sys.path.insert(0, '/home/jiayinliu/Desktop/Experiment/LowDimConditionalHGan/ControlledHamiltonianGAN_10/src')

import torch
from hgan.dataset import LipsonPendulumDataset

def test_dataset():
    print("Testing LipsonPendulumDataset...")
    
    # Test dataset initialization
    dataset = LipsonPendulumDataset(
        experiment_name="pend-real",
        save_dir="/home/jiayinliu/Desktop/Experiment/LowDimConditionalHGan/hamiltonian-nn/experiment-real",
        train=True,
        test_split=0.8,
        num_frames=16,
        delta=None,
    )
    
    print(f"✓ Dataset initialized")
    print(f"  Dataset size: {len(dataset)}")
    
    # Test __getitem__
    trajectory, mask, label_and_props = dataset[0]
    print(f"✓ __getitem__ works")
    print(f"  Trajectory shape: {trajectory.shape} (expected: (16, 2, 1))")
    print(f"  Mask shape: {mask.shape} (expected: (1,))")
    print(f"  Label and props shape: {label_and_props.shape} (expected: (0,))")
    
    assert trajectory.shape == (16, 2, 1), f"Wrong trajectory shape: {trajectory.shape}"
    assert mask.shape == (1,), f"Wrong mask shape: {mask.shape}"
    assert label_and_props.shape == (0,), f"Wrong label_and_props shape: {label_and_props.shape}"
    assert torch.all(mask == 1.0), "Mask should be all ones"
    
    # Test get_full_trajectory
    full_traj, full_mask = dataset.get_full_trajectory(train=True)
    print(f"✓ get_full_trajectory works")
    print(f"  Full trajectory shape: {full_traj.shape}")
    print(f"  Full mask shape: {full_mask.shape}")
    
    # Test find_best_match
    fake_traj = trajectory  # Use first sample as "fake"
    best_idx, best_rmse, all_rmses = dataset.find_best_match(fake_traj, full_traj)
    print(f"✓ find_best_match works")
    print(f"  Best index: {best_idx}")
    print(f"  Best RMSE: {best_rmse:.6f}")
    print(f"  Number of windows tested: {len(all_rmses)}")
    
    # The best match should be at index 0 with RMSE near 0 (since we're matching against itself)
    assert best_idx == 0, f"Best match should be at index 0, got {best_idx}"
    assert best_rmse < 0.01, f"RMSE should be near 0, got {best_rmse}"
    
    # Test plot_trajectory_comparison (save to temp file)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
    
    dataset.plot_trajectory_comparison(
        fake_traj=trajectory,
        real_traj=full_traj,
        save_path=temp_path,
        title="Test Plot",
        find_best_match=True
    )
    print(f"✓ plot_trajectory_comparison works")
    print(f"  Plot saved to: {temp_path}")
    
    # Clean up
    os.remove(temp_path)
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
    return True

if __name__ == "__main__":
    try:
        test_dataset()
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
