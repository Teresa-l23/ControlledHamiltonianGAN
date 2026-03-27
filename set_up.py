#!/usr/bin/env python3
"""
Set up SympNet training experiments for multiple physical systems.

Workflow:
1) Duplicate a template project folder for each system.
2) Generate one config per target generator frame count.
3) Force architecture=sympnet and set system-specific fields.

Then run training batches with run_sym.sh.
"""

import configparser
import os
import shutil


SYSTEMS = [
    {"system_name": "mass_spring", "folder_suffix": "MS"},
    {"system_name": "pendulum", "folder_suffix": "P"},
    {"system_name": "double_pendulum", "folder_suffix": "DP"},
    {"system_name": "two_body", "folder_suffix": "TB1"},
    {"system_name": "three_body", "folder_suffix": "THB2"},
]

GENERATOR_FRAMES = [30]
SYMPNET_N_LAYERS = 4


def create_configuration_file(
    source_config_path,
    target_config_path,
    system_name,
    folder_suffix,
    generator_frames,
    output_base_path,
):
    config = configparser.ConfigParser()
    config.read(source_config_path)

    if "experiment" not in config:
        config.add_section("experiment")
    if "video" not in config:
        config.add_section("video")
    if "paths" not in config:
        config.add_section("paths")

    config["experiment"]["architecture"] = "sympnet"
    config["experiment"]["system_name"] = system_name
    config["experiment"]["sympnet_n_layers"] = str(SYMPNET_N_LAYERS)
    config["video"]["generator_frames"] = str(generator_frames)

    output_dir_name = f"{folder_suffix}_sympnet_{generator_frames}"
    output_path = os.path.join(output_base_path, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    config["paths"]["input"] = output_path
    config["paths"]["output"] = output_path

    with open(target_config_path, "w") as configfile:
        config.write(configfile)

    return output_path


def create_system_folder(base_dir, source_folder, system_config):
    system_name = system_config["system_name"]
    folder_suffix = system_config["folder_suffix"]

    folder_name = f"ControlledHGAN_{folder_suffix}"
    target_folder = os.path.join(base_dir, folder_name)

    print(f"\n{'=' * 60}")
    print(f"Preparing system folder: {folder_name}")
    print(f"System: {system_name}")
    print(f"Generator frame variants: {GENERATOR_FRAMES}")
    print(f"{'=' * 60}")

    if os.path.exists(target_folder):
        print(f"Folder {target_folder} already exists. Reusing folder and refreshing configs...")
    else:
        print(f"Copying template folder from {source_folder} to {target_folder}...")
        shutil.copytree(
            source_folder,
            target_folder,
            symlinks=False,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".git",
                "logs",
                "*.log",
                "ControlledHGAN_*",
            ),
        )

    hgan_path = os.path.join(target_folder, "src", "hgan")
    output_base_path = os.path.join(hgan_path, "output")
    source_config_path = os.path.join(hgan_path, "configuration.ini")

    if not os.path.exists(source_config_path):
        raise FileNotFoundError(f"configuration.ini not found at: {source_config_path}")

    os.makedirs(output_base_path, exist_ok=True)

    for gen_frames in GENERATOR_FRAMES:
        config_filename = f"configuration_{gen_frames}.ini"
        target_config_path = os.path.join(hgan_path, config_filename)

        output_path = create_configuration_file(
            source_config_path=source_config_path,
            target_config_path=target_config_path,
            system_name=system_name,
            folder_suffix=folder_suffix,
            generator_frames=gen_frames,
            output_base_path=output_base_path,
        )

        print(f"  Created config file: {config_filename}")
        print(f"    - architecture: sympnet")
        print(f"    - system_name: {system_name}")
        print(f"    - generator_frames: {gen_frames}")
        print(f"    - sympnet_n_layers: {SYMPNET_N_LAYERS}")
        print(f"    - input/output: {output_path}")

    print(f"\nSystem folder '{folder_name}' is ready.")
    return True


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Prefer an explicit template folder if present, otherwise use current repo root.
    explicit_template = os.path.join(base_dir, "ControlledHamiltonianGAN")
    source_folder = explicit_template if os.path.exists(explicit_template) else base_dir

    print("\n" + "=" * 60)
    print("SympNet Experiment Setup Script")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"Template source folder: {source_folder}")

    if not os.path.exists(source_folder):
        print(f"\nError: Source folder not found: {source_folder}")
        return

    print(f"\nWill prepare {len(SYSTEMS)} system folders:")
    for system in SYSTEMS:
        print(f"  - ControlledHGAN_{system['folder_suffix']} ({system['system_name']})")
        print(
            f"    Configs: {', '.join([f'configuration_{frames}.ini' for frames in GENERATOR_FRAMES])}"
        )

    response = input("\nProceed with creating/updating system folders? (y/n): ")
    if response.lower() != "y":
        print("Aborted.")
        return

    success_count = 0
    for system_config in SYSTEMS:
        if create_system_folder(base_dir, source_folder, system_config):
            success_count += 1

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print(f"Successfully prepared {success_count}/{len(SYSTEMS)} system folders")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Run setup: python set_up.py")
    print("2. Run training batches: bash run_sym.sh")


if __name__ == "__main__":
    main()
