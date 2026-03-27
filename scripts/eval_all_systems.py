#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from hgan.configuration import load_config
from hgan.experiment import Experiment
from hgan.hgn.environments.environment_factory import EnvFactory
from hgan.hgn_datasets import all_systems_hgn, constant_physics_hgn


SYSTEMS: List[Tuple[str, str]] = [
    ("MS", "mass_spring"),
    ("P", "pendulum"),
    # ("DP", "double_pendulum"),
    ("TB1", "two_body"),
    ("THB2", "three_body"),
]

SYSTEM_PARTICLES = {
    "mass_spring": 1,
    "pendulum": 1,
    "double_pendulum": 2,
    "two_body": 2,
    "three_body": 3,
}


@dataclass
class EvalResult:
    system: str
    suffix: str
    status: str
    checkpoint_epoch: int = -1
    trajectory_mse_mean: float = float("nan")
    trajectory_mse_std: float = float("nan")
    pca_dim95: int = -1
    pca_pc1_ratio: float = float("nan")
    pca_pc2_ratio: float = float("nan")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_conditioning(experiment: Experiment, system_name: str, batch_size: int):
    system_index = all_systems_hgn.index(system_name)
    system_args = constant_physics_hgn[system_name]
    system_args = {
        key: (value() if not isinstance(value, list) else [_v() for _v in value])
        for key, value in system_args.items()
    }
    env_name = experiment.dataloader.dataset.system_name_mapping[system_name]
    system = EnvFactory.get_environment(env_name, **system_args)

    props = torch.tensor(
        system.physical_properties(vec_length=experiment.ndim_physics),
        dtype=torch.float32,
    )
    embedding = experiment.system_embedding(torch.tensor([system_index])).squeeze().float()
    label_and_props = torch.cat([embedding, props]).unsqueeze(0).repeat(batch_size, 1)
    label_and_props = label_and_props.to(experiment.device)

    num_particles = SYSTEM_PARTICLES[system_name]
    mask = torch.zeros(experiment.config.experiment.max_n, dtype=torch.float32)
    mask[:num_particles] = 1.0
    mask_batch = mask.unsqueeze(0).repeat(batch_size, 1).to(experiment.device)
    return system, system_args, env_name, label_and_props, mask, mask_batch


def compute_trajectory_mse(
    experiment: Experiment,
    system_name: str,
    num_samples: int,
    batch_size: int,
) -> Tuple[float, float]:
    system, system_args, env_name, label_and_props, mask_vec, mask_batch = build_conditioning(
        experiment, system_name, batch_size
    )

    mses: List[float] = []
    delta = 0.05
    mass = np.atleast_1d(system_args["mass"])
    n_frames = experiment.config.video.generator_frames

    while len(mses) < num_samples:
        fake_data = experiment.get_fake_data(
            n_frames=n_frames,
            label_and_props=label_and_props,
            mask=mask_batch,
        )
        fake_videos = fake_data["videos"].detach().cpu().numpy()  # [B, T, 2, N]

        take = min(batch_size, num_samples - len(mses))
        for sample in fake_videos[:take]:
            q0 = system.extract_q(sample, mask_vec.numpy(), frame_idx=0)
            q1 = system.extract_q(sample, mask_vec.numpy(), frame_idx=1)
            p0 = mass[:, None] * (q1 - q0) / delta

            if env_name == "NObjectGravity":
                real = system.calculate_fixed_rollout(q0, p0, n_frames, delta)
            else:
                real = system.calculate_fixed_rollout(q0.flatten(), p0.flatten(), n_frames, delta)

            valid_idx = np.where(mask_vec.numpy() > 0)[0]
            fake_valid = sample[:, :, valid_idx]
            real_valid = real[:, :, valid_idx]
            mses.append(float(np.mean((fake_valid - real_valid) ** 2)))

    return float(np.mean(mses)), float(np.std(mses))


def compute_pca_metrics(
    experiment: Experiment,
    system_name: str,
    latent_batch_size: int,
    output_prefix: str,
) -> Tuple[int, float, float]:
    _, _, _, label_and_props, _, _ = build_conditioning(experiment, system_name, latent_batch_size)

    Z, _, eps_motion = experiment.get_latent_sample(
        batch_size=latent_batch_size,
        n_frames=1,
        label_and_props=label_and_props,
    )
    trajectory, _ = experiment.rnn(
        torch.concat((label_and_props[0], eps_motion[0])).unsqueeze(0),
        n_frames=experiment.config.video.generator_frames,
    )
    trajectory = trajectory[:, 0, : experiment.ndim_q].detach().cpu().numpy().reshape(-1, experiment.ndim_q)
    x_train = Z[:, 0, : experiment.ndim_q].detach().cpu().numpy().reshape(-1, experiment.ndim_q)

    pca_full = PCA()
    pca_full.fit(x_train)
    evr = pca_full.explained_variance_ratio_
    cumsum = np.cumsum(evr)
    dim95 = int(np.argmax(cumsum >= 0.95) + 1)

    pca_2d = PCA(n_components=2)
    train_2d = pca_2d.fit_transform(x_train)
    traj_2d = pca_2d.transform(trajectory)

    ensure_dir(os.path.dirname(output_prefix))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    n_comp = min(20, len(evr))
    ax1.bar(np.arange(1, n_comp + 1), evr[:n_comp], color="#4e79a7")
    ax1.plot(np.arange(1, n_comp + 1), cumsum[:n_comp], color="#e15759", marker="o")
    ax1.set_title("PCA Spectrum")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained ratio / cumulative")
    ax1.grid(alpha=0.3)

    ax2.scatter(train_2d[:, 0], train_2d[:, 1], s=8, alpha=0.5, label="Initial latent q")
    ax2.scatter(traj_2d[:, 0], traj_2d[:, 1], s=10, alpha=0.7, marker="x", label="Trajectory q")
    ax2.set_title("PCA 2D Projection")
    ax2.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.1f}%)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_pca.png", bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "dim95": dim95,
        "pc1_ratio": float(pca_2d.explained_variance_ratio_[0]),
        "pc2_ratio": float(pca_2d.explained_variance_ratio_[1]),
    }
    with open(f"{output_prefix}_pca_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics["dim95"], metrics["pc1_ratio"], metrics["pc2_ratio"]


def find_latest_checkpoint_epoch(experiment: Experiment) -> int:
    saved_epochs = experiment.saved_epochs()
    if not saved_epochs:
        return -1
    return int(saved_epochs[-1])


def evaluate_system(
    base_dir: str,
    suffix: str,
    system_name: str,
    config_name: str,
    out_root: str,
    num_trajectory_samples: int,
    latent_batch_size: int,
    eval_batch_size: int,
) -> EvalResult:
    folder = os.path.join(base_dir, f"ControlledHGAN_{suffix}")
    config_path = os.path.join(folder, "src", "hgan", config_name)
    out_dir = os.path.join(out_root, system_name)
    ensure_dir(out_dir)

    if not os.path.exists(config_path):
        return EvalResult(system=system_name, suffix=suffix, status=f"missing_config:{config_path}")

    config = load_config(config_path)
    config.save(out_dir)
    experiment = Experiment(config)
    experiment.eval()

    latest_epoch = find_latest_checkpoint_epoch(experiment)
    if latest_epoch < 0:
        return EvalResult(system=system_name, suffix=suffix, status="no_checkpoint")

    device = "cpu" if not torch.cuda.is_available() else None
    experiment.load_epoch(latest_epoch, device=device)

    if eval_batch_size != experiment.batch_size:
        experiment.batch_size = eval_batch_size

    mse_mean, mse_std = compute_trajectory_mse(
        experiment=experiment,
        system_name=system_name,
        num_samples=num_trajectory_samples,
        batch_size=experiment.batch_size,
    )

    pca_prefix = os.path.join(out_dir, f"epoch_{latest_epoch:06d}")
    dim95, pc1, pc2 = compute_pca_metrics(
        experiment=experiment,
        system_name=system_name,
        latent_batch_size=latent_batch_size,
        output_prefix=pca_prefix,
    )

    return EvalResult(
        system=system_name,
        suffix=suffix,
        status="ok",
        checkpoint_epoch=latest_epoch,
        trajectory_mse_mean=mse_mean,
        trajectory_mse_std=mse_std,
        pca_dim95=dim95,
        pca_pc1_ratio=pc1,
        pca_pc2_ratio=pc2,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate all systems for trajectory MSE and PCA.")
    parser.add_argument("--base-dir", type=str, default="/home/jiayinliu/projects/SympNet")
    parser.add_argument("--config-name", type=str, default="configuration_60.ini")
    parser.add_argument("--output-dir", type=str, default="/home/jiayinliu/projects/SympNet/eval_results_all_systems")
    parser.add_argument("--num-trajectory-samples", type=int, default=100)
    parser.add_argument("--latent-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    return parser


def main():
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)

    all_results: List[EvalResult] = []
    for suffix, system in SYSTEMS:
        print(f"[eval] {system} ({suffix}) ...")
        result = evaluate_system(
            base_dir=args.base_dir,
            suffix=suffix,
            system_name=system,
            config_name=args.config_name,
            out_root=args.output_dir,
            num_trajectory_samples=args.num_trajectory_samples,
            latent_batch_size=args.latent_batch_size,
            eval_batch_size=args.eval_batch_size,
        )
        all_results.append(result)
        print(f"[eval] {system}: {result.status}")

    summary = [r.__dict__ for r in all_results]
    summary_path = os.path.join(args.output_dir, "summary_all_systems.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.output_dir, "summary_all_systems.csv")
    with open(csv_path, "w") as f:
        f.write(
            "system,suffix,status,checkpoint_epoch,trajectory_mse_mean,trajectory_mse_std,pca_dim95,pca_pc1_ratio,pca_pc2_ratio\n"
        )
        for r in all_results:
            f.write(
                f"{r.system},{r.suffix},{r.status},{r.checkpoint_epoch},{r.trajectory_mse_mean},"
                f"{r.trajectory_mse_std},{r.pca_dim95},{r.pca_pc1_ratio},{r.pca_pc2_ratio}\n"
            )

    print(f"[done] Wrote summary to {summary_path}")
    print(f"[done] Wrote table to {csv_path}")


if __name__ == "__main__":
    main()
