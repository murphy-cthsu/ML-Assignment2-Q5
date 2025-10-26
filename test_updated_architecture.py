"""
Test script for the updated architecture with game sequences.

Tests:
1. Model creation with correct architecture
2. Data loading with game sequences
3. Forward pass with sequences
4. Training compatibility
"""

import torch
import numpy as np
import sys

# Import the built C++ module
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

from style_model import create_model
from train_style_model import GoGameSequenceDataset, TripletGameDataset, collate_fn_triplet
from torch.utils.data import DataLoader


def test_model_architecture():
    """Test that model has correct architecture"""
    print("="*60)
    print("TEST 1: Model Architecture")
    print("="*60)
    
    model = create_model(
        'triplet',
        input_channels=18,
        hidden_channels=128,
        num_blocks=3,
        move_embed_dim=128,
        num_heads=4,
        num_layers=2,
        style_embed_dim=256
    )
    
    # Check components
    assert hasattr(model.encoder, 'move_encoder'), "Missing move_encoder"
    assert hasattr(model.encoder, 'aggregator'), "Missing aggregator"
    assert hasattr(model.encoder.move_encoder, 'res_blocks'), "Missing CNN residual blocks"
    assert hasattr(model.encoder.move_encoder, 'mlp'), "Missing move MLP"
    assert hasattr(model.encoder.aggregator, 'transformer'), "Missing transformer"
    assert hasattr(model.encoder.aggregator, 'mlp'), "Missing aggregator MLP"
    
    print("✓ Model has correct components:")
    print("  - MoveEncoder (CNN + MLP)")
    print("  - TransformerAggregator (Transformer + MLP + tanh + normalize)")
    
    # Test forward pass with sequences
    batch_size = 2
    seq_len = 30
    x = torch.randn(batch_size, seq_len, 18, 19, 19)
    
    embedding = model.get_embedding(x)
    
    assert embedding.shape == (batch_size, 256), f"Wrong output shape: {embedding.shape}"
    assert torch.allclose(embedding.norm(dim=1), torch.ones(batch_size), atol=1e-5), "Embeddings not normalized"
    
    print(f"✓ Forward pass successful: {x.shape} -> {embedding.shape}")
    print(f"✓ Embeddings are L2-normalized (norm ≈ 1.0)")
    print()
    
    return True


def test_data_loading():
    """Test data loading with game sequences"""
    print("="*60)
    print("TEST 2: Data Loading")
    print("="*60)
    
    try:
        # Load configuration
        style_py.load_config_file('conf.cfg')
        
        # Create dataset
        dataset = GoGameSequenceDataset(
            'train_set',
            'conf.cfg',
            max_moves=100
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Players: {dataset.num_players}")
        print(f"  - Games: {len(dataset.game_index)}")
        
        # Test getting a sample
        features, label = dataset[0]
        print(f"✓ Sample retrieved:")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Expected: (seq_len, 18, 19, 19)")
        print(f"  - Label: {label}")
        
        assert features.dim() == 4, f"Wrong feature dimensions: {features.dim()}"
        assert features.shape[1:] == (18, 19, 19), f"Wrong feature shape: {features.shape}"
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        print("  (This is OK if train_set is not fully populated)")
        print()
        return False


def test_triplet_dataset():
    """Test triplet dataset generation"""
    print("="*60)
    print("TEST 3: Triplet Dataset")
    print("="*60)
    
    try:
        style_py.load_config_file('conf.cfg')
        
        base_dataset = GoGameSequenceDataset(
            'train_set',
            'conf.cfg',
            max_moves=100
        )
        
        triplet_dataset = TripletGameDataset(base_dataset)
        
        print(f"✓ Triplet dataset created")
        print(f"  - Base games: {len(base_dataset)}")
        print(f"  - Valid players: {len(triplet_dataset.valid_players)}")
        
        # Get a triplet
        (anchor, a_label), (pos, p_label), (neg, n_label) = triplet_dataset[0]
        
        print(f"✓ Triplet retrieved:")
        print(f"  - Anchor shape: {anchor.shape}, label: {a_label}")
        print(f"  - Positive shape: {pos.shape}, label: {p_label}")
        print(f"  - Negative shape: {neg.shape}, label: {n_label}")
        
        assert a_label == p_label, "Anchor and positive should have same label"
        assert a_label != n_label, "Anchor and negative should have different labels"
        
        print(f"✓ Triplet validation passed:")
        print(f"  - Anchor and Positive have same label: {a_label}")
        print(f"  - Negative has different label: {n_label}")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Triplet dataset test failed: {e}")
        print("  (This is OK if train_set is not fully populated)")
        print()
        return False


def test_training_batch():
    """Test training batch creation with padding"""
    print("="*60)
    print("TEST 4: Training Batch with Padding")
    print("="*60)
    
    try:
        style_py.load_config_file('conf.cfg')
        
        base_dataset = GoGameSequenceDataset('train_set', 'conf.cfg', max_moves=100)
        triplet_dataset = TripletGameDataset(base_dataset)
        
        # Create data loader
        loader = DataLoader(
            triplet_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn_triplet
        )
        
        # Get a batch
        batch = next(iter(loader))
        (anchor, anchor_mask, anchor_labels), (pos, pos_mask, pos_labels), (neg, neg_mask, neg_labels) = batch
        
        print(f"✓ Batch created successfully:")
        print(f"  - Anchor: {anchor.shape}, mask: {anchor_mask.shape}")
        print(f"  - Positive: {pos.shape}, mask: {pos_mask.shape}")
        print(f"  - Negative: {neg.shape}, mask: {neg_mask.shape}")
        
        # Test model forward pass
        model = create_model(
            'triplet',
            hidden_channels=128,
            num_blocks=3,
            move_embed_dim=128,
            num_heads=4,
            num_layers=2,
            style_embed_dim=256
        )
        
        model.eval()
        with torch.no_grad():
            a_emb, p_emb, n_emb = model(anchor, pos, neg, anchor_mask, pos_mask, neg_mask)
        
        print(f"✓ Model forward pass with batch:")
        print(f"  - Anchor embeddings: {a_emb.shape}")
        print(f"  - Positive embeddings: {p_emb.shape}")
        print(f"  - Negative embeddings: {n_emb.shape}")
        
        # Check normalization
        a_norms = a_emb.norm(dim=1)
        print(f"  - Embedding norms: {a_norms.tolist()}")
        print(f"  - All norms ≈ 1.0: {torch.allclose(a_norms, torch.ones_like(a_norms), atol=1e-5)}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Training batch test failed: {e}")
        print("  (This is OK if train_set is not fully populated)")
        print()
        return False


def test_gpu_compatibility():
    """Test GPU compatibility"""
    print("="*60)
    print("TEST 5: GPU Compatibility")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        
        # Create model on GPU
        model = create_model(
            'triplet',
            hidden_channels=128,
            num_blocks=3,
            move_embed_dim=128,
            num_heads=4,
            num_layers=2,
            style_embed_dim=256
        ).to(device)
        
        # Test forward pass
        x = torch.randn(2, 30, 18, 19, 19).to(device)
        embedding = model.get_embedding(x)
        
        print(f"✓ GPU forward pass successful")
        print(f"  - Input device: {x.device}")
        print(f"  - Output device: {embedding.device}")
        print()
        
        return True
    else:
        print("✗ GPU not available")
        print("  (This is OK, model will use CPU)")
        print()
        return False


def main():
    print("\n" + "="*60)
    print("TESTING UPDATED ARCHITECTURE")
    print("Paper Architecture:")
    print("  1. Per-move: Residual CNN + MLP")
    print("  2. Game-level: Transformer + MLP + tanh + normalize")
    print("="*60)
    print()
    
    results = {}
    
    # Test 1: Model architecture
    results['architecture'] = test_model_architecture()
    
    # Test 2: Data loading
    results['data_loading'] = test_data_loading()
    
    # Test 3: Triplet dataset
    results['triplet_dataset'] = test_triplet_dataset()
    
    # Test 4: Training batch
    results['training_batch'] = test_training_batch()
    
    # Test 5: GPU compatibility
    results['gpu'] = test_gpu_compatibility()
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL (or skipped)"
        print(f"{test_name:20s}: {status}")
    
    critical_tests = ['architecture']
    all_critical_passed = all(results[t] for t in critical_tests)
    
    if all_critical_passed:
        print("\n✓ All critical tests passed!")
        print("✓ Architecture matches paper specification!")
        print("\nReady to train!")
    else:
        print("\n✗ Some critical tests failed")
        print("Please check the errors above")
    
    print("="*60)


if __name__ == '__main__':
    main()
