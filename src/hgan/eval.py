import os
import sys
import argparse
import time
import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hgan.configuration import load_config
from hgan.experiment import Experiment
from hgan.hgn_datasets import all_systems_hgn, constant_physics_hgn
from hgan.hgn.environments.environment_factory import EnvFactory

matplotlib.use("agg")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.ini specifying experiment parameters",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder where results will be generated",
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="Process every nth epoch checkpoint encountered (default 1)",
    )
    parser.add_argument(
        "--generated-videos-timeslots",
        type=int,
        default=8,
        help="Number of timeslots to include in generated videos pngs (default 8)",
    )
    parser.add_argument(
        "--generated-videos-samples",
        type=int,
        default=3,
        help="Number of samples to include in generated videos pngs (default 3)",
    )
    parser.add_argument(
        "--latent-batch-size",
        type=int,
        default=1024,
        help="Number of latent samples to generate for TSNE embedding (default 1024)",
    )
    parser.add_argument(
        "--calculate-fvd",
        dest="calculate_fvd",
        action="store_true",
        default=False,
        help="Calculate fvd score for every processed epoch (expensive operation!)",
    )
    parser.add_argument(
        "--fvd-batch-size",
        type=int,
        default=16,
        help="Number of real/fake videos to consider for fvd calculation (default (%default)s)",
    )
    parser.add_argument(
        "--fvd-on-cpu",
        action="store_true",
        default=False,
        help="Whether to run FVD on cpu (for low memory GPUs; default False)",
    )
    parser.add_argument(
        "--system_name",
        type=str,
        default="mass_spring",
        help="The system to run the eval on.",
    )
    parser.add_argument(
        "--override-output-folder",
        action="store_true",
        default=False,
        help="Whether to assume that the output folder is the same as the config path folder.",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        nargs='+',
        default=[2, 5, 30, 50, 100],
        help="Perplexity values for t-SNE visualization (default: 2 5 30 50 100)",
    )
    parser.add_argument(
        "--generate-trajectories",
        action="store_true",
        default=False,
        help="Generate HNN trajectories for comparison instead of normal evaluation",
    )
    parser.add_argument(
        "--num-trajectory-samples",
        type=int,
        default=100,
        help="Number of trajectory samples to generate per system (default: 100)",
    )
    parser.add_argument(
        "--trajectory-epoch",
        type=int,
        default=None,
        help="Specific epoch to use for trajectory generation (default: latest)",
    )
    return parser


def qualitative_results_img(
    experiment,
    png_path,
    fake=True,
    timeslots=8,
    samples=3,
    title="",
    epoch=None,
    save_video=False,
    label_and_props=None,
    colors=None,
):
    videos = (
        experiment.get_fake_data(label_and_props=label_and_props, colors=colors)
        if fake
        else experiment.get_real_data()
    )["videos"][
        :samples, ...
    ]  # (samples, nc, T, img_size, img_size)
    videos = videos.detach().cpu().numpy()

    # (samples, nc, T, img_size, img_size) => (samples, T, img_size, img_size, nc)
    videos = videos.transpose(0, 2, 3, 4, 1)

    if save_video:
        experiment.save_video(
            os.path.dirname(png_path),
            videos[0],
            epoch=epoch,
            prefix="fake_" if fake else "real_",
        )

    videos = videos[:samples, :timeslots, :, :, :]
    # Create a ndarray of video frames: timeslots, then samples_per_timeslot
    videos = videos.reshape((-1, *videos.shape[2:]))

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(title)
    # A grid in which each column represents a timeslot and each row a different instantiation of the
    # video at that time slot
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(samples, timeslots),
        axes_pad=0.1,
    )

    for ax, im in zip(grid, videos):
        ax.axis("off")
        ax.imshow(im)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.close(fig=fig)


def qualitative_results_latent(
    experiment,
    label_and_props,
    png_paths_prefix,
    perplexity_values=(2, 5, 30, 50, 100),
    title="",
    n_frames=30,
    projections=("tsne", "pca"),
    multi_system=False,
):
    batch_size = label_and_props.shape[0]
    Z, _, eps_motion = experiment.get_latent_sample(
        batch_size=batch_size, n_frames=1, label_and_props=label_and_props
    )  # shape (batch_size, n_frames, |ndim_q + ndim_p + ndim_content + ndim_label|, 1, 1)
    trajectory, _ = experiment.rnn(
        torch.concat((label_and_props[0], eps_motion[0])).unsqueeze(0),
        n_frames=n_frames,
    )
    trajectory = trajectory[:, 0, : experiment.ndim_q].data.cpu().numpy().squeeze()
    
    # Extract initial conditions from latent space
    X_train = Z[:, 0, : experiment.ndim_q].data.cpu().numpy().reshape(-1, experiment.ndim_q)
    X_traj = trajectory

    # Combine data for projection
    size_train = X_train.shape[0]
    X = np.vstack((X_train, X_traj))
    
    for projection_name in projections:
        if projection_name == "tsne":
            # Generate separate plots for each perplexity value
            for p in perplexity_values:
                fig, ax = plt.subplots(
                    figsize=(4, 6),
                    layout="constrained",
                    dpi=300,
                )
                projection = TSNE(n_components=2, perplexity=p, init="random")
                projected = projection.fit_transform(X)

                projected_train = projected[:size_train]
                projected_traj = projected[size_train:]
                
                ax.scatter(projected_train[:, 0], projected_train[:, 1], 
                         c='#3498db', marker='o', label="Initial conditions", 
                         alpha=0.6, s=25, edgecolors='none')
                ax.scatter(projected_traj[:, 0], projected_traj[:, 1], 
                         c='orange', marker='x', label="Trajectory", 
                         s=30, alpha=0.7)
                ax.legend(loc='upper right', frameon=True, fancybox=True, 
                        shadow=True, framealpha=0.9)
                ax.axis("off")
                
                # Save each perplexity plot separately with high quality
                png_path = f"{png_paths_prefix}_tsne_perplexity_{p}.png"
                os.makedirs(os.path.dirname(png_path), exist_ok=True)
                plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig=fig)
        else:  # PCA with eigenvalue analysis
            # Fit PCA on the full latent dimension to analyze all components
            pca_full = PCA()
            pca_full.fit(X_train)
            
            # Get eigenvalues (explained variance)
            eigenvalues = pca_full.explained_variance_
            explained_variance_ratio = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
            
            # Plot 1: Eigenvalue spectrum (scree plot)
            n_components = min(len(eigenvalues), 20)  # Show first 20 components
            ax1.bar(range(1, n_components + 1), eigenvalues[:n_components], 
                   color='#3498db', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Eigenvalue (Explained Variance)', fontsize=12, fontweight='bold')
            ax1.set_title('PCA Eigenvalue Spectrum', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Add cumulative variance on secondary y-axis
            ax1_twin = ax1.twinx()
            ax1_twin.plot(range(1, n_components + 1), cumulative_variance[:n_components] * 100,
                         color='#e74c3c', marker='o', linewidth=2, markersize=4, label='Cumulative')
            ax1_twin.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold', color='#e74c3c')
            ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
            ax1_twin.set_ylim([0, 105])
            
            # Plot 2: 2D PCA projection
            pca_2d = PCA(n_components=2)
            projected_train = pca_2d.fit_transform(X_train)
            projected_traj = pca_2d.transform(X_traj)
            
            ax2.scatter(projected_train[:, 0], projected_train[:, 1], 
                       c='#3498db', marker='o', label="Initial conditions", 
                       alpha=0.6, s=25, edgecolors='none')
            ax2.scatter(projected_traj[:, 0], projected_traj[:, 1], 
                       c='orange', marker='x', label="Trajectory", 
                       s=30, alpha=0.7)
            ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
            ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
            ax2.set_title('PCA 2D Projection', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.9)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Save PCA analysis plot
            png_path = f"{png_paths_prefix}_pca_eigenvalue_analysis.png"
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig=fig)
            
            # Save eigenvalue data to text file for further analysis
            eigenvalue_file = f"{png_paths_prefix}_pca_eigenvalues.txt"
            with open(eigenvalue_file, 'w') as f:
                f.write("PCA Eigenvalue Analysis\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total dimensions: {len(eigenvalues)}\n")
                f.write(f"First eigenvalue: {eigenvalues[0]:.6f}\n")
                f.write(f"Variance explained by PC1: {explained_variance_ratio[0]*100:.2f}%\n\n")
                
                # Find intrinsic dimensionality (95% variance threshold)
                dim_95 = np.argmax(cumulative_variance >= 0.95) + 1
                f.write(f"Intrinsic dimensionality (95% variance): {dim_95}\n")
                f.write(f"Cumulative variance with {dim_95} components: {cumulative_variance[dim_95-1]*100:.2f}%\n\n")
                
                f.write("Component-wise breakdown:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'PC':<6}{'Eigenvalue':<15}{'Var %':<12}{'Cumulative %':<15}\n")
                f.write("-" * 60 + "\n")
                for i in range(min(len(eigenvalues), 20)):
                    f.write(f"{i+1:<6}{eigenvalues[i]:<15.6f}{explained_variance_ratio[i]*100:<12.2f}{cumulative_variance[i]*100:<15.2f}\n")


def generate_hnn_trajectories_for_comparison(
    experiment,
    output_folder,
    system_name=None,
    num_samples=100,
    epoch=None,
    device=None,
    system_label_and_props=None,
    system_mask=None
):
    device = device or experiment.device
    
    trajectory_output_dir = os.path.join(output_folder, "hnn_trajectories")
    os.makedirs(trajectory_output_dir, exist_ok=True)
    
    if epoch is None:
        epoch = getattr(experiment, 'current_epoch', 'unknown')
    
    if system_name is None or system_name == "variable":
        systems_to_process = ["mass_spring", "pendulum", "double_pendulum", "two_body", "three_body"]
    else:
        systems_to_process = [system_name]
    
    for sys_name in systems_to_process:
        system_output_dir = os.path.join(trajectory_output_dir, sys_name)
        os.makedirs(system_output_dir, exist_ok=True)
        
        system_index = all_systems_hgn.index(sys_name)
        samples_generated = 0
        
        while samples_generated < num_samples:
            remaining_samples = num_samples - samples_generated
            samples_to_save = min(experiment.batch_size, remaining_samples)
            
            if system_label_and_props is not None:
                batch_label_and_props = system_label_and_props.repeat(experiment.batch_size, 1).to(device)
            else:
                batch_label_and_props = None
                
            if system_mask is not None:
                mask = system_mask.repeat(experiment.batch_size, 1).to(device)
            else:
                mask = None
            
            fake_data = experiment.get_fake_data(
                n_frames=experiment.config.video.generator_frames,
                label_and_props=batch_label_and_props,
                mask=mask
            )
            
            fake_videos = fake_data["videos"]
            
            for i in range(samples_to_save):
                sample_idx = samples_generated + i
                sample_trajectory = fake_videos[i].detach().cpu().numpy()
                
                save_path = os.path.join(system_output_dir, f"hnn_{sys_name}_{sample_idx:06d}_data.npz")
                np.savez(
                    save_path,
                    fake=sample_trajectory,
                    real=sample_trajectory,
                    mask=system_mask,
                    system_name=experiment.dataloader.dataset.system_name_mapping[sys_name],
                    epoch=epoch,
                    system_index=system_index
                )
                            
            samples_generated += samples_to_save


def main(*args):

    device = "cpu" if not torch.cuda.is_available() else None

    args = get_parser().parse_args(args)
    config = load_config(args.config_path)

    output_folder = args.output_folder
    config.save(output_folder)

    experiment = Experiment(config)
    if args.override_output_folder:
        experiment.config.paths.output = os.path.dirname(args.config_path)
    experiment.eval()
    
    if args.generate_trajectories:
        saved_epochs = experiment.saved_epochs()
        if not saved_epochs:
            return
        
        target_epoch = args.trajectory_epoch if args.trajectory_epoch else saved_epochs[-1]
        experiment.load_epoch(target_epoch, device=device)
        
        system_particles_map = {
            "mass_spring": 1,
            "pendulum": 1, 
            "double_pendulum": 2,
            "two_body": 2,
            "three_body": 3
        }
        
        system_label_and_props = None
        mask = None
        if args.system_name and args.system_name != "variable":
            system_index = all_systems_hgn.index(args.system_name)
            system_args = constant_physics_hgn[args.system_name]
            system_args = {
                k: (v() if not isinstance(v, list) else [_v() for _v in v])
                for k, v in system_args.items()
            }
            system = EnvFactory.get_environment(
                experiment.dataloader.dataset.system_name_mapping[args.system_name],
                **system_args,
            )
            props = torch.tensor(system.physical_properties(vec_length=experiment.ndim_physics))
            
            system_embedding = experiment.system_embedding(torch.tensor([system_index])).squeeze()
            system_label_and_props = torch.cat([system_embedding, props]).unsqueeze(0)
            
            num_particles = system_particles_map[args.system_name]
            mask = torch.zeros(config.experiment.max_n, dtype=torch.float32)
            mask[:num_particles] = 1.0

        generate_hnn_trajectories_for_comparison(
            experiment=experiment,
            output_folder=output_folder,
            system_name=args.system_name,
            num_samples=args.num_trajectory_samples,
            epoch=target_epoch,
            device=device,
            system_label_and_props=system_label_and_props,
            system_mask=mask
        )
        return
    
    # Original evaluation code continues...
    batch_size = args.latent_batch_size

    # Check if we need to handle multiple systems
    is_multi_system = args.system_name == "variable"
    all_system_names = ["mass_spring", "pendulum", "double_pendulum", "two_body", "three_body"]

    if not is_multi_system:
        # Single system handling as before
        system_index = all_systems_hgn.index(args.system_name)
        system_args = constant_physics_hgn[args.system_name]
        system_args = {
            k: (v() if not isinstance(v, list) else [_v() for _v in v])
            for k, v in system_args.items()
        }
        system = EnvFactory.get_environment(
            experiment.dataloader.dataset.system_name_mapping[args.system_name],
            **system_args,
        )
        props = torch.tensor(system.physical_properties(vec_length=experiment.ndim_physics))
    else:
        # Pre-compute all system properties for multi-system mode
        multi_props = []
        for sys_name in all_system_names:
            sys_args = constant_physics_hgn[sys_name]
            sys_args = {
                k: (v() if not isinstance(v, list) else [_v() for _v in v])
                for k, v in sys_args.items()
            }
            sys_env = EnvFactory.get_environment(
                experiment.dataloader.dataset.system_name_mapping[sys_name],
                **sys_args,
            )
            sys_props = torch.tensor(sys_env.physical_properties(vec_length=experiment.ndim_physics))
            
            samples_per_system = batch_size // 5
            if sys_name == all_system_names[-1]: 
                samples_per_system = batch_size - 4 * samples_per_system
            
            sys_props_batch = sys_props.unsqueeze(0).repeat(samples_per_system, 1)
            multi_props.append(sys_props_batch)
        
        # Concatenate all system properties
        multi_props = torch.cat(multi_props, dim=0)

    saved_epochs = experiment.saved_epochs()
    for epoch in saved_epochs[:: args.every_nth]:
        experiment.load_epoch(epoch, device=device)
        
        if is_multi_system:
            multi_labels = []
            for sys_name in all_system_names:
                sys_index = all_systems_hgn.index(sys_name)
                sys_embedding = experiment.system_embedding(torch.tensor([sys_index])).squeeze()
                samples_per_system = batch_size // 5
                if sys_name == all_system_names[-1]:
                    samples_per_system = batch_size - 4 * samples_per_system
                sys_label_batch = sys_embedding.unsqueeze(0).repeat(samples_per_system, 1)
                multi_labels.append(sys_label_batch)
            multi_labels = torch.cat(multi_labels, dim=0)
            label_and_props = torch.cat([multi_labels, multi_props], dim=1).to(experiment.device)
        else:
            label_and_props = torch.cat(
                (
                    experiment.system_embedding(torch.tensor([system_index])).squeeze(),
                    props,
                )
            )
            label_and_props = (
                label_and_props.unsqueeze(0)
                .repeat(batch_size, 1)
                .to(experiment.device)
            )  # (batch_size, ndim_label + ndim_physics)

        # Z, _, _ = experiment.get_latent_sample(
        #     batch_size=config.experiment.batch_size,
        #     n_frames=config.video.generator_frames,
        #     label_and_props=label_and_props,
        # )
        # Z_motion = Z[0, :, : experiment.ndim_epsilon, :, :].squeeze()
        # hnn_input = torch.concat(  # Note order: label_props, then Z_motion
        #     (
        #         label_and_props[0]
        #         .unsqueeze(0)
        #         .repeat(config.video.generator_frames, 1),
        #         Z_motion,
        #     ),
        #     axis=1,
        # )
        # energy = experiment.rnn.hnn(hnn_input)
        # std_energy = float(torch.std(energy.squeeze()))

        # with open(os.path.join(output_folder, "energy.txt"), "a") as f:
        #     f.write(f"epoch={epoch}, std_energy={std_energy}\n")

        # logger.info("  Generating Videos Image")
        # qualitative_results_img(
        #     experiment,
        #     f"{output_folder}/videos_{epoch:06d}.png",
        #     timeslots=args.generated_videos_timeslots,
        #     samples=args.generated_videos_samples,
        #     title=f"Epoch {epoch}",
        #     epoch=epoch,
        #     save_video=True,
        #     fake=True,
        #     label_and_props=label_and_props,
        #     colors=colors,
        # )

        # For the first label_and_props sampled 1024 times from the latent space,
        # plot the TSNE embedding of the q part of the latent space
        qualitative_results_latent(
            experiment=experiment,
            label_and_props=label_and_props,
            png_paths_prefix=f"{output_folder}/config_{epoch:06d}",
            perplexity_values=args.perplexity,
            title=f"Epoch {epoch}",
            multi_system=is_multi_system,
        )

        if args.calculate_fvd:
            fvd_device = "cpu" if args.fvd_on_cpu else experiment.device
            fvd = experiment.fvd(device=fvd_device, max_videos=args.fvd_batch_size)
            fvd_score_file = os.path.join(output_folder, "fvd_scores.txt")
            with open(fvd_score_file, "a") as f:
                f.write(f"epoch={epoch}, fvd={fvd}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
