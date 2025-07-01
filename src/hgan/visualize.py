import numpy as np
import os
from hgan.hgn.environments.environment_factory import EnvFactory


def estimate_momentum(traj, frame_idx, delta_t=1.0, mass=1.0):
    v = (traj[frame_idx + 1] - traj[frame_idx]) / delta_t
    return mass * v

def prepare_init_state(pos, mom):
    """
    拼接 [x1, y1, p1x, p1y, x2, y2, p2x, p2y, ...]
    pos, mom: [2, N]
    return: [4 * N]
    """
    N = pos.shape[1]
    init_state = []
    for i in range(N):
        init_state.extend([
            pos[0, i], pos[1, i],
            mom[0, i], mom[1, i]
        ])
    return np.array(init_state, dtype=np.float32)

def generate_real_traj_from_fake(
    fake_traj,
    system_name,
    system_args,
    frame_idx=9,
    delta_t=0.05,
    mass=1.0,
    save_path='real_traj.npy',
    num_steps=30,
    t_span=(0, 10)
):
    system = EnvFactory.get_environment(system_name, **system_args)
    
    pos = fake_traj[frame_idx]       # [2, N]
    mom = estimate_momentum(fake_traj, frame_idx, delta_t, mass)  # [2, N]

    init_state = prepare_init_state(pos, mom)  # shape: [4 * N]

    env = EnvFactory.get_environment(system_name, **system_args)
    real_traj = env.simulate(init_state, t_span, num_steps)

    # 保存真实轨迹
    np.save(save_path, real_traj)
    print(f"Saved real trajectory to {save_path}")

    return real_traj, init_state