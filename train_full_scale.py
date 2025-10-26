#!/usr/bin/env python
"""
Full-scale training script with resume capability and advanced features.
"""

import os
import argparse
import json
import torch
from pathlib import Path

def get_config(preset='medium'):
    """Get training configuration presets"""
    configs = {
        'test': {
            'hidden_channels': 64,
            'num_blocks': 2,
            'move_embed_dim': 64,
            'num_heads': 4,
            'num_transformer_layers': 2,
            'style_embed_dim': 128,
            'epochs': 5,
            'batch_size': 4,
            'num_samples': 50,
            'max_moves': 30,
        },
        'small': {
            'hidden_channels': 64,
            'num_blocks': 2,
            'move_embed_dim': 64,
            'num_heads': 4,
            'num_transformer_layers': 2,
            'style_embed_dim': 128,
            'epochs': 50,
            'batch_size': 32,
            'num_samples': 1000,
            'max_moves': 50,
        },
        'medium': {
            'hidden_channels': 128,
            'num_blocks': 3,
            'move_embed_dim': 128,
            'num_heads': 8,
            'num_transformer_layers': 3,
            'style_embed_dim': 256,
            'epochs': 50,
            'batch_size': 16,
            'num_samples': 1000,
            'max_moves': 75,
        },
        'large': {
            'hidden_channels': 256,
            'num_blocks': 5,
            'move_embed_dim': 256,
            'num_heads': 8,
            'num_transformer_layers': 4,
            'style_embed_dim': 512,
            'epochs': 100,
            'batch_size': 8,
            'num_samples': 2000,
            'max_moves': 100,
        },
    }
    
    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(configs.keys())}")
    
    return configs[preset]


def main():
    parser = argparse.ArgumentParser(description='Full-scale Go Style Detection Training')
    
    # Preset selection
    parser.add_argument('--preset', type=str, default='medium',
                       choices=['test', 'small', 'medium', 'large'],
                       help='Configuration preset (default: medium)')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, default='test_set/query_set',
                       help='Training data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: models/full_training_<preset>)')
    parser.add_argument('--config_file', type=str, default='conf.cfg',
                       help='Configuration file for C++ backend')
    
    # Training control
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_interval', type=int, default=10,
                       help='Evaluate every N epochs (if validation set provided)')
    
    # Override preset values (optional)
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override preset: number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override preset: batch size')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Override preset: samples per epoch')
    parser.add_argument('--max_moves', type=int, default=None,
                       help='Override preset: maximum moves per game')
    
    args = parser.parse_args()
    
    # Get preset configuration
    config = get_config(args.preset)
    
    # Override with command line arguments if provided
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_samples is not None:
        config['num_samples'] = args.num_samples
    if args.max_moves is not None:
        config['max_moves'] = args.max_moves
    
    # Set output directory
    if args.output_dir is None:
        output_dir = f'models/full_training_{args.preset}'
    else:
        output_dir = args.output_dir
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("=" * 70)
    print("Go Style Detection - Full-scale Training")
    print("=" * 70)
    print(f"\nPreset: {args.preset}")
    print(f"\nData:")
    print(f"  Training directory: {args.train_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Config file: {args.config_file}")
    
    print(f"\nTraining Parameters:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Samples per epoch: {config['num_samples']}")
    print(f"  Max moves per game: {config['max_moves']}")
    
    print(f"\nModel Architecture:")
    print(f"  Hidden channels: {config['hidden_channels']}")
    print(f"  Residual blocks: {config['num_blocks']}")
    print(f"  Move embedding dim: {config['move_embed_dim']}")
    print(f"  Transformer heads: {config['num_heads']}")
    print(f"  Transformer layers: {config['num_transformer_layers']}")
    print(f"  Style embedding dim: {config['style_embed_dim']}")
    
    # Estimate model parameters
    approx_params = (
        config['hidden_channels'] * 18 * 3 * 3 +  # Initial conv
        config['num_blocks'] * config['hidden_channels']**2 * 18 +  # Residual blocks (approx)
        config['move_embed_dim'] * config['hidden_channels'] +  # MLP
        config['num_transformer_layers'] * config['move_embed_dim']**2 * 4 +  # Transformer (approx)
        config['style_embed_dim'] * config['move_embed_dim']  # Final MLP
    )
    print(f"  Estimated parameters: ~{approx_params / 1e6:.2f}M")
    
    print(f"\nSystem:")
    print(f"  Workers: {args.num_workers}")
    print(f"  Save interval: {args.save_interval} epochs")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
    
    print("=" * 70)
    print()
    
    # Save configuration
    config_save = {
        'preset': args.preset,
        'train_dir': args.train_dir,
        'config_file': args.config_file,
        **config,
        'num_workers': args.num_workers,
        'save_interval': args.save_interval,
    }
    
    config_path = os.path.join(output_dir, 'full_training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    print(f"Configuration saved to: {config_path}\n")
    
    # Build training command
    cmd = [
        'python', 'train_style_model.py',
        '--train_dir', args.train_dir,
        '--config_file', args.config_file,
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--num_samples', str(config['num_samples']),
        '--hidden_channels', str(config['hidden_channels']),
        '--num_blocks', str(config['num_blocks']),
        '--move_embed_dim', str(config['move_embed_dim']),
        '--num_heads', str(config['num_heads']),
        '--num_transformer_layers', str(config['num_transformer_layers']),
        '--style_embed_dim', str(config['style_embed_dim']),
        '--max_moves', str(config['max_moves']),
        '--output_dir', output_dir,
        '--num_workers', str(args.num_workers),
        '--save_interval', str(args.save_interval),
    ]
    
    if args.resume:
        cmd.extend(['--resume', args.resume])
    
    # Execute training
    import subprocess
    print("Starting training...\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print(f"Best model saved to: {output_dir}/best_model.pth")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Training failed!")
        print("=" * 70)
        return result.returncode
    
    return 0


if __name__ == '__main__':
    exit(main())
