import os
import glob
import skvideo.io
from skimage.transform import resize
import numpy as np
import torch
from torch.utils.data import Dataset
import jax
import functools
from hgan.dm_hamiltonian_dynamics_suite import datasets
from hgan.configuration import config
from hgan.dm_datasets import all_systems, constant_physics, variable_physics
from hgan.hgn_datasets import (
    all_systems_hgn,
    constant_physics_hgn,
    variable_physics_hgn,
)
from hgan.hgn.environments.environment_factory import EnvFactory
import matplotlib.pyplot as plt


class AviDataset(Dataset):
    def __init__(self, datapath, T):
        self.T = T
        self.datapath = os.path.join(datapath, "resized_data")
        self.files = glob.glob(os.path.join(self.datapath, "*"))

        self.videos = self.get_videos()
        self.n_videos = len(self.videos)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        start = np.random.randint(0, video.shape[1] - (self.T + 1))
        end = start + self.T
        return video[:, start:end, ...].astype(np.float32)

    def get_videos(self):
        videos = [skvideo.io.vread(file) for file in self.files]
        # transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
        videos = [video.transpose(3, 0, 1, 2) / 255.0 for video in videos]

        return videos


class ToyPhysicsDataset(Dataset):
    def __init__(self, datapath, delta=1, train=True, resize=True, normalize=True):
        train_test = "train" if train else "test"
        self.T = config.video.frames
        self.resize = resize
        self.normalize = normalize

        self.delta = delta
        self.datapath = os.path.join(datapath, train_test)
        self.files = glob.glob(os.path.join(self.datapath, "*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.datapath, f"{idx:06}.npy")

        vid = np.load(filename)
        vid = vid[:: self.delta]  # orig dt is 0.05
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - (self.T + 1))
        end = start + self.T
        vid = vid[start:end]

        vid = (
            np.asarray(
                [
                    resize(
                        img,
                        (config.experiment.img_size, config.experiment.img_size, nc),
                    )
                    for img in vid
                ]
            )
            if self.resize
            else vid
        )
        # vid = np.asarray([resize(img, (96, 96, nc)) for img in vid])
        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)
        # normalize -1 1
        vid = (vid - 0.5) / 0.5 if self.normalize else vid
        # vid = (vid - 0.5)/0.5

        return vid.astype(np.float32)


class ToyPhysicsDatasetNPZ(Dataset):
    def __init__(self, *, datapath, num_frames, delta=1, train=True):
        train_test = "train" if train else "test"
        self.num_frames = num_frames
        self.delta = delta
        self.datapath = os.path.join(datapath, train_test)
        self.files = glob.glob(os.path.join(self.datapath, "*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.datapath, str(idx).zfill(5) + ".npz")

        vid = np.load(filename)["arr_0"]
        vid = vid[:: self.delta]  # orig dt is 0.05
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - (self.num_frames + 1))
        end = start + self.num_frames
        vid = vid[start:end]

        if img_size != config.experiment.img_size:
            vid = np.asarray(
                [
                    resize(
                        img,
                        (config.experiment.img_size, config.experiment.img_size, nc),
                    )
                    for img in vid
                ]
            )

        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)

        if config.video.normalize:
            vid = (vid - 0.5) / 0.5

        return vid.astype(np.float32), torch.tensor([])


class RealtimeDataset(Dataset):
    def __init__(
        self,
        *,
        ndim_physics=10,
        system_name=None,
        num_frames=16,
        delta=0.05,
        train=True,
        system_physics_constant=True,
        system_color_constant=True,
        system_friction=False,
        total_frames=100,
        img_size=32,
        normalize=False,
    ):

        jax.config.update("jax_enable_x64", True)

        if system_name is None:
            system_names = all_systems
            if system_color_constant:
                system_names = [x for x in system_names if "_COLORS" not in x]
            if system_friction:
                system_names = [x for x in system_names if "_FRICTION" in x]
        else:
            system_name = system_name.upper()
            if not system_color_constant:
                system_name += "_COLORS"
            if system_friction:
                system_name += "_FRICTION"

            assert system_name in all_systems, f"Unknown system {system_name}"
            system_names = [system_name]

        assert system_names, "No system selected"
        self.system_names = system_names

        self.num_frames = num_frames
        self.total_frames = total_frames
        self.delta = delta
        self.train = train
        self.ndim_physics = ndim_physics
        self.img_size = img_size
        self.normalize = normalize

        self.generate_fn = {}  # Generate functions, keyed by system
        self.features = {}  # Fixed features across all trajectories, keyed by system

        for system_name in self.system_names:
            cls, config_ = getattr(datasets, system_name)
            config_dict = config_()

            # Tweak the physics parameters to our liking
            # TODO: Is this okay to do for speedup? Will this modify the characteristics of the experiment drastically?
            config_dict["image_resolution"] = img_size
            _physics_key = (
                system_name.replace("COLORS", "").replace("FRICTION", "").rstrip("_")
            )
            if system_physics_constant:
                config_dict |= constant_physics[_physics_key]
            else:
                config_dict |= variable_physics[_physics_key]

            obj = cls(**config_dict)

            f = functools.partial(
                datasets.generate_sample,
                system=obj,
                dt=0.05,  # Blanchette 2021
                # num_steps is always 1 less than the no. of samples we wish to generate
                num_steps=total_frames - 1,
                steps_per_dt=1,
            )

            self.generate_fn[system_name] = f
            self.features[system_name] = f(0)["other"]

    def __len__(self):
        return 50_000 if self.train else 10_000  # Blanchette 2021

    def _physics_vector_from_data(self, data):
        ndim_physics = self.ndim_physics
        if ndim_physics <= 0:
            return 0

        props = np.zeros((ndim_physics,))

        i = 0
        # note: dicts are ordered in py >= 3.7 so we have a deterministic order
        for k, v in data["other"].items():
            v = v.squeeze()
            if v.size == 1:
                props[i] = v.item()
                i += 1
                if i >= len(props):
                    return props
            else:
                for _v in v:
                    props[i] = _v
                    i += 1
                    if i >= len(props):
                        return props

        return props

    def __getitem__(self, item):
        system_name_index = np.random.choice(len(self.system_names))
        system_name = self.system_names[system_name_index]

        data = self.generate_fn[system_name](item)  # num_steps + 1, L, L, num_channels
        image = data["image"]
        # assert isinstance(image, jax.numpy.ndarray)
        # assert image.dtype == np.uint8

        vid = np.array(image / 255)
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - self.num_frames + 1)
        end = start + self.num_frames
        vid = vid[start:end]

        if img_size != self.img_size:
            vid = np.asarray(
                [
                    resize(
                        img,
                        (
                            self.img_size,
                            self.img_size,
                            nc,
                        ),
                    )
                    for img in vid
                ]
            )

        # transpose each video to (nc, n_frames, img_size, img_size)
        vid = vid.transpose(3, 0, 1, 2)

        if self.normalize:
            vid = (vid - 0.5) / 0.5

        props = self._physics_vector_from_data(data)
        return vid.astype(np.float32), system_name_index, props


class HGNRealtimeDataset(Dataset):
    def __init__(
        self,
        *,
        ndim_label=3,
        ndim_physics=10,
        ndim_color=0,
        system_name=None,
        num_frames=16,
        delta=0.05,
        train=True,
        system_physics_constant=True,
        system_color_constant=True,
        system_friction=False,
        total_frames=100,
        img_size=32,
        normalize=False,
    ):

        self.system_names = all_systems_hgn
        self.n_systems = len(self.system_names)
        if system_name is None:
            self.system_index = None
        else:
            self.system_index = self.system_names.index(system_name)

        self.num_frames = num_frames
        self.total_frames = total_frames
        self.delta = delta
        self.train = train
        self.ndim_label = ndim_label
        self.ndim_physics = ndim_physics
        self.ndim_color = ndim_color
        self.img_size = img_size
        self.normalize = normalize

        assert not bool(system_friction), "No friction supported yet"

        self.system_physics_constant = system_physics_constant
        self.system_color_constant = system_color_constant
        self.system_friction = system_friction

        self.system_name_mapping = {
            "mass_spring": "Spring",
            "pendulum": "Pendulum",
            "double_pendulum": "ChaoticPendulum",
            "two_body": "NObjectGravity",
            "three_body": "NObjectGravity",
        }

        self.system_embedding = torch.nn.Embedding(self.n_systems, self.ndim_label)

    def __len__(self):
        return 50_000 if self.train else 10_000  # Blanchette 2021
    
    def compute_rmse(self, traj_gen, traj_real, eps=1e-8):
        traj_gen = np.array(traj_gen)
        traj_real = np.array(traj_real)

        diff = traj_gen - traj_real
        rmse = np.sqrt(np.mean(diff ** 2))

        return rmse
    
    def plot_2d_trajectory_comparison(self, traj, mask, real, save_path, label1='Fake', label2='Real', system_name=None):
        import os
        valid_idx = np.where(mask > 0)[0]
        N_eff = len(valid_idx)
        T = traj.shape[0]
        T_real = real.shape[0]

        # 优先用传入的 system_name，否则用 self.system_name
        sys_name = system_name if system_name is not None else getattr(self, 'system_name', 'Unknown')

        plt.figure(figsize=(6, 6))
        colors = plt.cm.tab10.colors
        rmse_list = []
        if sys_name == "Spring":
            # 1D系统，所有粒子画在一张图上，x随时间变化
            time = np.arange(T)
            for i, idx in enumerate(valid_idx):
                plt.plot(time, traj[:, 0, idx], label=f'{label1} Particle {idx+1}', color=colors[i % len(colors)], linestyle='-')
                plt.plot(time, real[:, 0, idx], label=f'{label2} Particle {idx+1}', color=colors[i % len(colors)], linestyle='--')
                fake_traj = traj[:, 0, idx]
                real_traj = real[:, 0, idx]
                if T == T_real:
                    rmse = self.compute_rmse(fake_traj, real_traj, eps=1e-8)
                    rmse_list.append(rmse)
            plt.xlabel('Time')
            plt.ylabel('x')
        else:
            # 2D系统，所有粒子画在一张图上，xy轨迹
            for i, idx in enumerate(valid_idx):
                plt.plot(traj[:, 0, idx], traj[:, 1, idx], label=f'{label1} Particle {idx+1}', color=colors[i % len(colors)], linestyle='-')
                plt.plot(real[:, 0, idx], real[:, 1, idx], label=f'{label2} Particle {idx+1}', color=colors[i % len(colors)], linestyle='--')
                fake_traj = traj[:, :, idx]
                real_traj = real[:, :, idx]
                if T == T_real:
                    rmse = self.compute_rmse(fake_traj, real_traj, eps=1e-8)
                    rmse_list.append(rmse)
            plt.xlabel('x')
            plt.ylabel('y')

        if T == T_real and rmse_list:
            mean_rmse = np.mean(rmse_list)
            title = f"{sys_name} (RMSE={mean_rmse:.4f})"
        else:
            title = f"{sys_name} (T mismatch)"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Save trajectory data
        if system_name is None:
            save_data_path = os.path.splitext(save_path)[0] + '_data.npz'
            np.savez(save_data_path, fake=traj, real=real, mask=mask, system_name=sys_name)

    def _compute_momentum(self, q_t, q_t1):
        mass = np.atleast_1d(self.system_args["mass"])
        p = mass[:, None] * (q_t1 - q_t) / self.delta
        return p.reshape(q_t.shape)

    def pad_traj(self, traj, n_max=3):
        n_particles = traj.shape[2]
        T = traj.shape[0]
        padded = torch.zeros((T, 2, n_max), dtype=torch.float32)
        padded[:, :, :n_particles] = torch.tensor(traj, dtype=torch.float32)

        mask = torch.zeros(n_max, dtype=torch.float32)
        mask[:n_particles] = 1.0

        return padded, mask

    def __getitem__(self, item):
        if self.system_index is None:
            system_index = np.random.choice(self.n_systems)
        else:
            system_index = self.system_index

        system_name = self.system_names[system_index]

        system_args_which = {True: constant_physics_hgn, False: variable_physics_hgn}[
            self.system_physics_constant
        ][system_name]

        self.system_args = {
            k: (v() if not isinstance(v, list) else [_v() for _v in v])
            for k, v in system_args_which.items()
        }

        self.system_name = self.system_name_mapping[system_name]
        system = EnvFactory.get_environment(self.system_name, **self.system_args)
        # We're not using self.total_frames here at all, since we only want self.num_frames from
        # the rollout, and the rollouts are randomly initialized anyway.

        colors = None
        rollout = None
        # Rollouts are not guaranteed to give us self.num_frames in certain
        # cases where solve_ivp fails - keep trying till they do.
        while rollout is None or rollout.shape[0] != self.num_frames:
            rollout = system.sample_random_rollouts(
                number_of_frames=self.num_frames,
                delta_time=self.delta,
                number_of_rollouts=1,
                img_size=self.img_size,
                noise_level=0.1,
                radius_bound="auto",
                color=True,
                seed=None,
                constant_color=self.system_color_constant,
            )
        padded, mask = self.pad_traj(rollout)

        labels_and_props = torch.cat(
            (
                self.system_embedding(torch.tensor([system_index])).squeeze(),
                torch.tensor(system.physical_properties(vec_length=self.ndim_physics)),
            )
        )

        return padded, mask, labels_and_props
    
    def get_system_ids(self, label_and_props):
        """
        label_and_props: torch.Tensor (B, n_label+params)
        返回 system_id: (B,) 整数ID
        """
        system_code = label_and_props[:, :3]
        unique_codes = {tuple(code): idx for idx, code in enumerate(np.unique(system_code, axis=0))}
        system_id = np.array([unique_codes[tuple(code)] for code in system_code])
        return system_id
    


    def comparison(self, trajectory, mask, folder, epoch, prefix, frame_idx=0):
        os.makedirs(folder, exist_ok=True)
        filename = f"{prefix}{epoch:0>6}"
        file_path = os.path.join(folder, f"{filename}.jpg")

        system = EnvFactory.get_environment(self.system_name, **self.system_args)
        q = system.extract_q(trajectory, mask, frame_idx=frame_idx)
        q1 = system.extract_q(trajectory, mask, frame_idx=frame_idx+1)
        p = self._compute_momentum(q,q1)
        
        rollout = system.calculate_fixed_rollout(q, p, number_of_frames=self.num_frames, delta_time=self.delta)

        self.plot_2d_trajectory_comparison(trajectory, mask, rollout, file_path)
   
class LipsonPendulumDataset(Dataset):
    """
    Dataset for real/simulated pendulum data from Lipson et al. 2009.
    Processes data using the same approach as hamiltonian-nn/experiment-real/data.py
    """
    def __init__(
        self,
        *,
        experiment_name="pend-real",  # or "pend-sim"
        save_dir=None,
        train=True,
        test_split=0.8,
        num_frames=16,
        delta=None,
    ):
        """
        Args:
            experiment_name: "pend-real" for real pendulum data, "pend-sim" for simulated
            save_dir: Directory to save/load the dataset zip file
            train: Not used (kept for compatibility) - uses all data for generative model
            test_split: Not used (kept for compatibility)
            num_frames: Number of consecutive frames to sample
            delta: Time step between frames (if None, uses original spacing)
        """
        self.experiment_name = experiment_name
        self.train = train
        self.num_frames = num_frames
        self.delta = delta
        
        # Set default save_dir if not provided
        if save_dir is None:
            save_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "hamiltonian-nn",
                "experiment-real"
            )
        self.save_dir = save_dir
        
        # Load and process data - use all data for generative model
        self.data = self._get_dataset()
        
        # Use all data for training (no train/test split for generative model)
        self.x = self.data['x']
        self.dx = self.data['dx']
        self.t = self.data['t']
    
    def _read_lipson(self, dataset_name, zip_dir):
        """Read dataset from Lipson zip file"""
        import zipfile
        zip_path = os.path.join(zip_dir, 'invar_datasets.zip')
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(f'{dataset_name}.txt') as f:
                return f.read().decode('utf-8')
    
    def _str2array(self, data_str):
        """Convert string data to numpy array"""
        lines = data_str.strip().split('\n')
        # Parse header line - remove '%' and extra spaces
        header_names = lines[0].strip().replace('%', '').split()
        # Add 'trial' and 't' to match the original format
        # The actual data columns are: trial, time, and then the header variables
        dnames = ['d' + n for n in header_names]
        names = ['trial', 't'] + header_names + dnames
        
        data = []
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                vals = [float(x) for x in line.split()]
                data.append(vals)
        return np.array(data), names
    
    def _get_dataset(self):
        """Load and process Lipson pendulum dataset - use all data for generative model"""
        if self.experiment_name == "pend-sim":
            dataset_name = "pendulum_h_1"
        elif self.experiment_name == "pend-real":
            dataset_name = "real_pend_h_1"
        else:
            raise ValueError(f"Unknown experiment name: {self.experiment_name}")
        
        os.makedirs(self.save_dir, exist_ok=True)
        out_file = os.path.join(self.save_dir, 'invar_datasets.zip')
        
        # Check if zip file exists in save_dir, if not check parent directory
        if not os.path.exists(out_file):
            parent_zip = os.path.join(os.path.dirname(self.save_dir), 'invar_datasets.zip')
            if os.path.exists(parent_zip):
                out_file = parent_zip
 
        
        # Use the directory containing the zip file for read_lipson
        zip_dir = os.path.dirname(out_file)
        data_str = self._read_lipson(dataset_name, zip_dir)
        state, names = self._str2array(data_str)
        
        # Put data in a dictionary structure
        data = {k: state[:, i:i+1] for i, k in enumerate(names)}
        data['x'] = state[:, 2:4]
        data['dx'] = (data['x'][1:] - data['x'][:-1]) / (data['t'][1:] - data['t'][:-1])
        data['x'] = data['x'][:-1]
        
        return data
    
    def __len__(self):
        return max(0, len(self.x) - self.num_frames + 1)
    
    def __getitem__(self, idx):
        """
        Returns trajectory format compatible with CHGAN framework:
            trajectory: (num_frames, 2, 1) - [q, p] for each timestep, reshaped for single particle
            mask: (1,) - all ones since we have 1 particle
            label_and_props: empty tensor for compatibility
        """
        if idx + self.num_frames > len(self.x):
            idx = len(self.x) - self.num_frames
        
        x = self.x[idx:idx + self.num_frames]  # (T, 2) where columns are [q, p]
        
        # Reshape to (T, 2, 1) to match trajectory format
        trajectory = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        
        # Single particle, so mask is all ones
        mask = torch.ones(1, dtype=torch.float32)
        
        # Empty label and props for unconditional training
        label_and_props = torch.tensor([], dtype=torch.float32)
        
        return trajectory, mask, label_and_props
    
    def get_full_trajectory(self, train=None):
        """
        Get the full trajectory for evaluation.
        Args:
            train: Not used (kept for compatibility) - returns all data
        Returns:
            trajectory: (T, 2, 1) full trajectory
            mask: (1,) all ones
        """
        # Use all data for generative model
        x = self.x
        trajectory = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        mask = torch.ones(1, dtype=torch.float32)
        
        return trajectory, mask
    
    def compute_rmse(self, traj_gen, traj_real):
        """
        Compute RMSE between two trajectories.
        Args:
            traj_gen: (T, 2, 1) generated trajectory
            traj_real: (T, 2, 1) real trajectory
        Returns:
            rmse: scalar RMSE value
        """
        traj_gen = np.array(traj_gen)
        traj_real = np.array(traj_real)
        diff = traj_gen - traj_real
        rmse = np.sqrt(np.mean(diff ** 2))
        return rmse
    
    def find_best_match(self, fake_traj, real_traj=None):
        """
        Find the best matching subsequence in the real trajectory for the fake trajectory.
        Uses sliding window with RMSE metric across the entire trajectory.
        
        Args:
            fake_traj: (T_fake, 2, 1) generated trajectory
            real_traj: (T_real, 2, 1) real trajectory (if None, uses full trajectory)
        
        Returns:
            best_idx: starting index of best match in real trajectory
            best_rmse: RMSE at best match
            all_rmses: array of RMSE values for each window position
        """
        if real_traj is None:
            # Use full trajectory for comparison
            real_traj, _ = self.get_full_trajectory()
        
        fake_traj = np.array(fake_traj)
        real_traj = np.array(real_traj)
        
        T_fake = fake_traj.shape[0]
        T_real = real_traj.shape[0]
        
        if T_fake > T_real:
            return 0, float('inf'), np.array([float('inf')])
        
        # Compute RMSE for each possible window
        rmses = []
        for start_idx in range(T_real - T_fake + 1):
            window = real_traj[start_idx:start_idx + T_fake]
            rmse = self.compute_rmse(fake_traj, window)
            rmses.append(rmse)
        
        rmses = np.array(rmses)
        best_idx = np.argmin(rmses)
        best_rmse = rmses[best_idx]
        
        return best_idx, best_rmse, rmses
    
    def plot_trajectory_comparison(self, fake_traj, real_traj=None, save_path=None, 
                                   title=None, find_best_match=True):
        """
        Plot comparison between fake and real trajectories.
        
        Args:
            fake_traj: (T_fake, 2, 1) generated trajectory
            real_traj: (T_real, 2, 1) real trajectory (if None, uses full trajectory)
            save_path: path to save the plot
            title: plot title
            find_best_match: if True, find best matching window in real_traj
        """
        fake_traj = np.array(fake_traj).squeeze()  # (T_fake, 2)
        
        if real_traj is None:
            # Use full trajectory for comparison
            real_traj, _ = self.get_full_trajectory()
        
        real_traj = np.array(real_traj).squeeze()  # (T_real, 2)
        
        # Find best match if requested
        if find_best_match and fake_traj.shape[0] <= real_traj.shape[0]:
            best_idx, best_rmse, _ = self.find_best_match(
                fake_traj.reshape(-1, 2, 1), 
                real_traj.reshape(-1, 2, 1)
            )
            real_traj = real_traj[best_idx:best_idx + fake_traj.shape[0]]
            match_info = f" (Best match at t={best_idx}, RMSE={best_rmse:.4f})"
        else:
            # Just use first frames of real trajectory
            T_fake = fake_traj.shape[0]
            real_traj = real_traj[:T_fake]
            rmse = self.compute_rmse(
                fake_traj.reshape(-1, 2, 1),
                real_traj.reshape(-1, 2, 1)
            )
            match_info = f" (RMSE={rmse:.4f})"
        
        # Create 2-subplot figure: phase space and time series
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Phase space plot (q vs p)
        axes[0].plot(fake_traj[:, 0], fake_traj[:, 1], 'b-', label='Fake', linewidth=2)
        axes[0].plot(real_traj[:, 0], real_traj[:, 1], 'r--', label='Real', linewidth=2)
        axes[0].set_xlabel('Position (q)')
        axes[0].set_ylabel('Momentum (p)')
        axes[0].set_title('Phase Space')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Time series plot
        time = np.arange(fake_traj.shape[0])
        axes[1].plot(time, fake_traj[:, 0], 'b-', label='Fake q', linewidth=2)
        axes[1].plot(time, real_traj[:, 0], 'r--', label='Real q', linewidth=2)
        axes[1].set_xlabel('Time step')
        axes[1].set_ylabel('Position (q)')
        axes[1].set_title('Position vs Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        if title is None:
            title = f"Lipson Pendulum Trajectory Comparison{match_info}"
        else:
            title = f"{title}{match_info}"
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def comparison(self, trajectory, mask, folder, epoch, prefix, frame_idx=0):
        """
        Comparison method compatible with HGNRealtimeDataset.
        For Lipson pendulum, we plot the trajectory with best match from full trajectory.
        
        Args:
            trajectory: (T, 2, 1) fake trajectory
            mask: (1,) mask (not used for single particle)
            folder: output folder
            epoch: training epoch
            prefix: filename prefix
            frame_idx: not used for Lipson (kept for compatibility)
        """
        os.makedirs(folder, exist_ok=True)
        filename = f"{prefix}{epoch:0>6}"
        file_path = os.path.join(folder, f"{filename}.jpg")
        
        # Get full trajectory for comparison (all data)
        real_trajectory, _ = self.get_full_trajectory()
        real_trajectory_np = real_trajectory.numpy()
        
        # Plot with best match
        self.plot_trajectory_comparison(
            trajectory,
            real_trajectory_np,
            save_path=file_path,
            title=f"Epoch {epoch}",
            find_best_match=True
        )
   