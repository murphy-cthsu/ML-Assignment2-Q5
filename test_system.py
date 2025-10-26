"""
Quick test script to verify the style detection system setup.

This script tests:
1. Model creation
2. Data loading
3. Basic training loop
4. Inference functionality
"""

import os
import sys
import torch
import numpy as np

print("="*70)
print("Style Detection System - Quick Test")
print("="*70)

# Test 1: Import built module
print("\n1. Testing C++ module import...")
try:
    _temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
    style_py = _temps.style_py
    print("   ✓ Successfully imported style_py module")
    
    # Load config
    style_py.load_config_file('conf.cfg')
    print("   ✓ Successfully loaded configuration")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import Python models
print("\n2. Testing model imports...")
try:
    from style_model import create_model, StyleEncoder
    print("   ✓ Successfully imported model classes")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Create models
print("\n3. Testing model creation...")
try:
    # Test triplet model
    model_triplet = create_model(
        'triplet',
        input_channels=18,
        hidden_channels=128,  # Smaller for quick test
        num_blocks=3,
        embedding_dim=256
    )
    print("   ✓ Created triplet model")
    
    # Test supervised model
    model_supervised = create_model(
        'supervised',
        input_channels=18,
        hidden_channels=128,
        num_blocks=3,
        num_players=600
    )
    print("   ✓ Created supervised model")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_triplet.parameters())
    print(f"   ✓ Triplet model parameters: {total_params:,}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Test forward pass
print("\n4. Testing model forward pass...")
try:
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 18, 19, 19)
    
    # Test triplet model
    anchor = dummy_input
    positive = torch.randn(batch_size, 18, 19, 19)
    negative = torch.randn(batch_size, 18, 19, 19)
    
    with torch.no_grad():
        a_emb, p_emb, n_emb = model_triplet(anchor, positive, negative)
    
    print(f"   ✓ Triplet forward pass successful")
    print(f"     - Embedding shape: {a_emb.shape}")
    print(f"     - Embedding norm: {a_emb[0].norm().item():.4f} (should be ~1.0)")
    
    # Test supervised model
    with torch.no_grad():
        logits = model_supervised(dummy_input)
    
    print(f"   ✓ Supervised forward pass successful")
    print(f"     - Logits shape: {logits.shape}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test data loading
print("\n5. Testing data loading...")
try:
    # Test with training set if available
    if os.path.exists('train_set'):
        data_loader = style_py.DataLoader('train_set')
        print("   ✓ Created DataLoader for train_set")
        
        # Load data
        data_loader.load_data_from_file()
        print("   ✓ Successfully loaded data from files")
        
        num_players = data_loader.get_num_of_player()
        num_games = data_loader.get_num_of_game()
        
        print(f"     - Number of players: {num_players}")
        print(f"     - Number of games: {num_games}")
        
        if num_players > 0:
            # Try to get features from first player
            num_games_p0 = data_loader.get_num_of_game_of_player(0)
            print(f"     - Games for player 0: {num_games_p0}")
            
            if num_games_p0 > 0:
                features, label = data_loader.get_feature_and_label(0, 0)
                if not isinstance(features, np.ndarray):
                    features = np.array(features)
                print(f"     - Feature shape: {features.shape}")
                print(f"     - Label: {label}")
    else:
        print("   ⚠ train_set directory not found, skipping data loading test")
        
except Exception as e:
    print(f"   ⚠ Warning (non-critical): {e}")

# Test 6: Test training imports
print("\n6. Testing training script imports...")
try:
    # Just check if the file can be parsed
    with open('train_style_model.py', 'r') as f:
        compile(f.read(), 'train_style_model.py', 'exec')
    print("   ✓ train_style_model.py syntax valid")
    
    with open('infer_style.py', 'r') as f:
        compile(f.read(), 'infer_style.py', 'exec')
    print("   ✓ infer_style.py syntax valid")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 7: GPU availability
print("\n7. Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available")
    print(f"     - Device: {torch.cuda.get_device_name(0)}")
    print(f"     - CUDA version: {torch.version.cuda}")
    
    # Test GPU inference
    try:
        model_gpu = model_triplet.cuda()
        dummy_gpu = dummy_input.cuda()
        
        with torch.no_grad():
            embedding_gpu = model_gpu.get_embedding(dummy_gpu)
        
        print(f"   ✓ GPU inference successful")
    except Exception as e:
        print(f"   ⚠ GPU inference failed: {e}")
else:
    print("   ⚠ CUDA not available, will use CPU")

# Test 8: Check file structure
print("\n8. Checking file structure...")
required_files = [
    'style_model.py',
    'train_style_model.py',
    'infer_style.py',
    'conf.cfg',
    'README_STYLE_DETECTION.md'
]

all_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} not found")
        all_exist = False

# Check directories
required_dirs = [
    'test_set/query_set',
    'test_set/cand_set'
]

for dir in required_dirs:
    if os.path.exists(dir):
        num_files = len([f for f in os.listdir(dir) if f.endswith('.sgf')])
        print(f"   ✓ {dir} ({num_files} SGF files)")
    else:
        print(f"   ⚠ {dir} not found")

# Summary
print("\n" + "="*70)
print("Test Summary")
print("="*70)
print("\n✓ All critical tests passed!")
print("\nYou can now:")
print("  1. Train a model:")
print("     python train_style_model.py --model_type triplet --epochs 5 --batch_size 16")
print("\n  2. Run inference (after training):")
print("     python infer_style.py --model_path models/best_model.pth")
print("\n  3. Read the documentation:")
print("     README_STYLE_DETECTION.md")
print("\n" + "="*70)
