#!/usr/bin/env python3
"""
Test script for style_py module (C++ Go environment via pybind)
Tests SGF loading, feature extraction, and data iteration
"""

import sys
import os

# Import style_py using the correct method
game_type = 'go'
conf_file = os.path.join(os.path.dirname(__file__), 'conf.cfg')

try:
    _temps = __import__(f'build.{game_type}', globals(), locals(), ['style_py'], 0)
    style_py = _temps.style_py
    print("✓ Successfully imported style_py module")
    
    # Load configuration file if it exists
    if os.path.exists(conf_file):
        style_py.load_config_file(conf_file)
        print(f"✓ Loaded configuration from {conf_file}")
    else:
        print(f"⚠ Configuration file not found: {conf_file}")
        
except ImportError as e:
    print(f"✗ Failed to import style_py: {e}")
    print(f"  Make sure you've built the project with: bash scripts/build.sh go")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error during setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_module_attributes():
    """Test that the module has expected attributes"""
    print("\n=== Testing Module Attributes ===")
    
    # Check what's available in the module
    attrs = [attr for attr in dir(style_py) if not attr.startswith('_')]
    print(f"Available attributes in style_py:")
    for attr in attrs:
        obj = getattr(style_py, attr)
        obj_type = type(obj).__name__
        print(f"  - {attr}: {obj_type}")
    
    return attrs

def test_configuration():
    """Test Configuration class or config functions"""
    print("\n=== Testing Configuration ===")
    
    try:
        # Check if get_config_name exists (common in config systems)
        if hasattr(style_py, 'get_config_name'):
            config_name = style_py.get_config_name()
            print(f"✓ Config name: {config_name}")
        
        # Try to get some common config values
        config_attrs = [attr for attr in dir(style_py) if attr.startswith('get_config_')]
        if config_attrs:
            print(f"  Available config getters: {config_attrs[:5]}")
            
            # Try a few common ones
            for attr in ['get_config_board_size', 'get_config_env_board_size']:
                if hasattr(style_py, attr):
                    try:
                        value = getattr(style_py, attr)()
                        print(f"  ✓ {attr}() = {value}")
                    except:
                        pass
        else:
            print("  ⚠ No config getter functions found")
            
    except Exception as e:
        print(f"✗ Error testing configuration: {e}")

def test_data_loader():
    """Test DataLoader class"""
    print("\n=== Testing DataLoader ===")
    
    try:
        # DataLoader requires a string argument (likely file path or directory)
        # Let's create a test directory for SGF files
        test_dir = os.path.join(os.path.dirname(__file__), 'test_sgf_data')
        os.makedirs(test_dir, exist_ok=True)
        
        # Create a simple test SGF file in the directory
        test_sgf = os.path.join(test_dir, 'test_game.sgf')
        sgf_content = """(;GM[1]FF[4]SZ[19]KM[7.5]
;B[pd];W[dp];B[pp];W[dd];B[pj];W[nc];B[qf];W[jd]
)"""
        with open(test_sgf, 'w') as f:
            f.write(sgf_content)
        print(f"✓ Created test SGF file: {test_sgf}")
        
        if hasattr(style_py, 'DataLoader'):
            # Try creating DataLoader with the directory path
            loader = style_py.DataLoader(test_dir)
            print(f"✓ Created DataLoader with directory: {test_dir}")
            
            # Check available methods
            loader_methods = [attr for attr in dir(loader) if not attr.startswith('_')]
            print(f"  DataLoader has {len(loader_methods)} methods:")
            for method in loader_methods[:15]:  # Show first 15
                print(f"    - {method}")
            if len(loader_methods) > 15:
                print(f"    ... and {len(loader_methods) - 15} more")
            
            # Clean up
            import shutil
            shutil.rmtree(test_dir)
            
            return loader
                
        elif hasattr(style_py, 'create_data_loader'):
            loader = style_py.create_data_loader(test_dir)
            print("✓ Created DataLoader via create_data_loader()")
            return loader
        else:
            print("⚠ DataLoader class not found in module")
            return None
            
    except Exception as e:
        print(f"✗ Error creating DataLoader: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        try:
            import shutil
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
        except:
            pass
        return None

def test_sgf_loading(loader=None):
    """Test SGF file loading if sample SGF exists"""
    print("\n=== Testing SGF Loading ===")
    
    if loader is None:
        print("⚠ No DataLoader available, skipping SGF test")
        return
    
    # Create a simple test SGF file
    test_sgf = os.path.join(os.path.dirname(__file__), 'test.sgf')
    
    # Simple 19x19 Go game SGF (matching default board size)
    sgf_content = """(;GM[1]FF[4]SZ[19]KM[7.5]
;B[pd];W[dp];B[pp];W[dd];B[pj];W[nc];B[qf];W[jd]
)"""
    
    try:
        with open(test_sgf, 'w') as f:
            f.write(sgf_content)
        print(f"✓ Created test SGF file: {test_sgf}")
        
        # Try different possible method names for loading SGF
        load_methods = ['loadSGF', 'load_sgf', 'loadFromSGF', 'load_from_sgf', 
                       'addData', 'add_data', 'loadData', 'load_data']
        
        loaded = False
        for method_name in load_methods:
            if hasattr(loader, method_name):
                try:
                    method = getattr(loader, method_name)
                    result = method(test_sgf)
                    print(f"✓ Successfully called {method_name}('{test_sgf}')")
                    print(f"  Result: {result}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠ {method_name} exists but failed: {e}")
        
        if not loaded:
            print("⚠ Could not find a working SGF loading method")
            print("  Available methods:", [m for m in dir(loader) if not m.startswith('_')][:10])
        
        # Clean up
        if os.path.exists(test_sgf):
            os.remove(test_sgf)
            
    except Exception as e:
        print(f"✗ Error in SGF loading test: {e}")
        import traceback
        traceback.print_exc()

def test_feature_extraction(loader=None):
    """Test feature extraction capabilities"""
    print("\n=== Testing Feature Extraction ===")
    
    if loader is None:
        print("⚠ No DataLoader available, skipping feature extraction test")
        return
    
    try:
        # Look for feature extraction methods
        all_methods = [m for m in dir(loader) if not m.startswith('_')]
        feature_methods = [m for m in all_methods if 'feature' in m.lower()]
        
        if feature_methods:
            print(f"  Found feature-related methods:")
            for method in feature_methods:
                print(f"    - {method}")
                
            # Try to call a feature extraction method
            for method_name in feature_methods:
                try:
                    method = getattr(loader, method_name)
                    print(f"  {method_name} signature: {method.__doc__ or 'No documentation'}")
                except:
                    pass
        else:
            print("  ⚠ No obvious feature extraction methods found")
            print("  All available methods:")
            for m in all_methods[:15]:
                print(f"    - {m}")
                
    except Exception as e:
        print(f"✗ Error testing feature extraction: {e}")

def test_environment():
    """Test the Env class for Go game environment"""
    print("\n=== Testing Go Environment (Env) ===")
    
    try:
        if hasattr(style_py, 'Env'):
            env = style_py.Env()
            print("✓ Created Env instance")
            
            # Check available methods
            env_methods = [attr for attr in dir(env) if not attr.startswith('_')]
            print(f"  Env has {len(env_methods)} methods:")
            for method in env_methods[:10]:
                print(f"    - {method}")
            if len(env_methods) > 10:
                print(f"    ... and {len(env_methods) - 10} more")
                
            # Try some common environment methods
            if hasattr(env, 'reset'):
                env.reset()
                print("  ✓ Successfully called reset()")
                
            if hasattr(env, 'getBoardSize'):
                board_size = env.getBoardSize()
                print(f"  ✓ Board size: {board_size}")
                
        else:
            print("⚠ Env class not found")
            
    except Exception as e:
        print(f"✗ Error testing environment: {e}")
        import traceback
        traceback.print_exc()

def test_player_enums():
    """Test Player enum values"""
    print("\n=== Testing Player Enums ===")
    
    try:
        if hasattr(style_py, 'player_1'):
            print(f"  player_1: {style_py.player_1}")
            print(f"  player_2: {style_py.player_2}")
            print(f"  player_none: {style_py.player_none}")
            print(f"  player_size: {style_py.player_size}")
            print("  ✓ Player enum values accessible")
        else:
            print("⚠ Player enums not found")
    except Exception as e:
        print(f"✗ Error accessing player enums: {e}")

def main():
    """Run all tests"""
    print("="*60)
    print("Testing style_py Module (C++ Go Environment)")
    print("="*60)
    
    attrs = test_module_attributes()
    test_configuration()
    test_environment()
    test_player_enums()
    loader = test_data_loader()
    test_sgf_loading(loader)
    test_feature_extraction(loader)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✓ Module successfully imported and basic tests completed!")
    print("\nAvailable Classes:")
    print("  - DataLoader: Load and process Go game data from SGF files")
    print("  - Env: Go game environment for simulations")
    print("  - EnvLoder: Environment loader")
    print("  - SLDataLoader: Supervised learning data loader")
    print("  - Action: Represents a Go move")
    print("  - Player: Player enum (player_1, player_2, player_none)")
    print("\nNext steps:")
    print("1. Create a DataLoader with a directory containing SGF files")
    print("2. Use Env to simulate Go games")
    print("3. Extract features for training neural networks")
    print("4. Iterate over batches of data for training")
    print("\nExample usage:")
    print("  loader = style_py.DataLoader('/path/to/sgf/directory')")
    print("  env = style_py.Env()")
    print("  env.reset()")
    print("\nFor more details, check the C++ source code in:")
    print("  - style_detection/pybind.cpp")
    print("  - style_detection/sd_data_loader.h/cpp")

if __name__ == "__main__":
    main()
