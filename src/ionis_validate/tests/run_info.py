#!/usr/bin/env python3
"""
run_info.py — IONIS V22-gamma System and Model Information

Displays model version, checkpoint details, PhysicsOverrideLayer config,
system configuration, and installed package paths.

Usage:
  python run_info.py
  ionis-validate info
"""

import json
import os
import platform
import sys

import torch

from ionis_validate.model import IonisGate, get_device
from ionis_validate.physics_override import PhysicsOverrideLayer
from ionis_validate import _data_path


def main():
    # Load config
    config_path = _data_path("config_v22.json")
    if not os.path.exists(config_path):
        print(f"  ERROR: Config not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path) as f:
        config = json.load(f)

    checkpoint_path = _data_path(config["checkpoint"])
    device = get_device()

    print()
    print("=" * 60)
    print("  IONIS V22-gamma — System & Model Information")
    print("=" * 60)

    # Model info
    print("\n  MODEL")
    print(f"  {'─' * 50}")
    print(f"  Version:       {config['version']}-{config.get('variant', '')} ({config['phase']})")
    print(f"  Architecture:  {config['model']['architecture']}")
    print(f"  DNN dim:       {config['model']['dnn_dim']}")
    print(f"  Hidden dim:    {config['model']['hidden_dim']}")
    print(f"  Input dim:     {config['model']['input_dim']}")
    print(f"  Sidecar hidden: {config['model']['sidecar_hidden']}")

    # Parameter count
    model = IonisGate(
        dnn_dim=config["model"]["dnn_dim"],
        sidecar_hidden=config["model"]["sidecar_hidden"],
        sfi_idx=config["model"]["sfi_idx"],
        kp_penalty_idx=config["model"]["kp_penalty_idx"],
        gate_init_bias=config["model"].get("gate_init_bias"),
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:    {param_count:,}")

    # Physics Override
    override = PhysicsOverrideLayer()
    print(f"\n  PHYSICS OVERRIDE")
    print(f"  {'─' * 50}")
    print(f"  {override.describe()}")

    # Checkpoint info
    print(f"\n  CHECKPOINT")
    print(f"  {'─' * 50}")
    print(f"  Path:          {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        size_bytes = os.path.getsize(checkpoint_path)
        if size_bytes > 1024 * 1024:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{size_bytes / 1024:.1f} KB"
        print(f"  File size:     {size_str}")
        print(f"  Format:        safetensors")

        # Load metadata from companion JSON
        meta_path = checkpoint_path.replace(".safetensors", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

            epoch = metadata.get("epoch")
            val_pearson = metadata.get("val_pearson")
            val_rmse = metadata.get("val_rmse")
            date_range = metadata.get("date_range")
            sample_size = metadata.get("sample_size")
            tst_900 = metadata.get("tst_900_score")
            ki7mt = metadata.get("ki7mt_hard_pass")

            if epoch is not None:
                print(f"  Epoch:         {epoch}")
            if val_pearson is not None:
                print(f"  Pearson:       {val_pearson:+.4f}")
            if val_rmse is not None:
                print(f"  RMSE:          {val_rmse:.4f} sigma")
            if tst_900:
                print(f"  TST-900:       {tst_900}")
            if ki7mt:
                print(f"  KI7MT:         {ki7mt}")
            if date_range:
                print(f"  Date range:    {date_range}")
            if sample_size:
                print(f"  Sample size:   {sample_size:,}")
    else:
        print(f"  Status:        NOT FOUND")

    # Features
    features = config.get("features", [])
    if features:
        print(f"\n  FEATURES ({len(features)})")
        print(f"  {'─' * 50}")
        for i, feat in enumerate(features):
            print(f"  [{i:>2d}] {feat}")

    # System info
    print(f"\n  SYSTEM")
    print(f"  {'─' * 50}")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  Device:        {device}")
    print(f"  Platform:      {platform.system()} {platform.machine()}")
    print(f"  Hostname:      {platform.node()}")

    if device.type == "cuda":
        print(f"  CUDA version:  {torch.version.cuda}")
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print(f"  MPS:           available")

    # Install paths
    print(f"\n  INSTALL PATHS")
    print(f"  {'─' * 50}")
    print(f"  Config:        {config_path}")
    print(f"  Tests:         {os.path.dirname(os.path.abspath(__file__))}")
    print(f"  Package:       {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
