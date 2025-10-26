 # Build Environment Setup and Testing Progress

This document outlines the complete journey from setting up the build environment to successfully testing the built Go module.

---

## 📋 Table of Contents

1. [Initial Challenges](#initial-challenges)
2. [Environment Setup](#environment-setup)
3. [Dependency Resolution](#dependency-resolution)
4. [Build Configuration Fixes](#build-configuration-fixes)
5. [Compilation Issues](#compilation-issues)
6. [Linking Problems](#linking-problems)
7. [Testing the Built Module](#testing-the-built-module)
8. [Final Results](#final-results)

---

## 🚧 Initial Challenges

### Problem: Missing Torch (PyTorch C++)
**Error:**
```
CMake Error: By not providing "FindTorch.cmake" in CMAKE_MODULE_PATH
Could not find a package configuration file provided by "Torch"
```

**Root Cause:** PyTorch C++ library (libtorch) not installed

---

## 🔧 Environment Setup

### Step 1: Create Conda Environment

```bash
# Created dedicated conda environment for the project
conda create -n ml_hw2 python=3.10
conda activate ml_hw2
```

**Why:** Isolated environment prevents dependency conflicts with other projects

### Step 2: Install Core Dependencies

```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install additional dependencies
conda install boost opencv cmake
pip install pybind11
```

**Versions Installed:**
- PyTorch: 2.x with CUDA 12.4 support
- Boost: 1.82.0
- OpenCV: 4.10.0
- CMake: 4.1.2
- pybind11: 2.12.1 (downgraded from 2.13.6)

---

## 🔍 Dependency Resolution

### Issue 1: pybind11 Compatibility

**Error:**
```
CMake Error: Unknown CMake command "python_add_library"
```

**Solution:**
```bash
# Downgraded pybind11 to fix CMake compatibility
pip uninstall -y pybind11
pip install "pybind11<2.13"
```

**Why:** pybind11 2.13.6 had compatibility issues with CMake 4.1.2

### Issue 2: Python Development Headers

**Added to CMakeLists.txt:**
```cmake
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
```

**Why:** pybind11 requires Python development headers for building bindings

---

## 🛠️ Build Configuration Fixes

### 1. Added Policy for Compatibility

**File:** `Q5/CMakeLists.txt`
```cmake
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()
```

**Why:** Ensures modern CMake behavior and reduces warnings

### 2. Disabled ALE (Arcade Learning Environment)

**Modified Files:**
- `Q5/CMakeLists.txt`
- `Q5/minizero/CMakeLists.txt`

**Changes:**
```cmake
# find_package(ale REQUIRED)  # Only needed for Atari games
```

**Why:** Building for Go game type, not Atari; ALE not needed

### 3. Excluded Atari and Chess Source Files

**File:** `Q5/minizero/minizero/environment/CMakeLists.txt`
```cmake
file(GLOB_RECURSE SRCS *.cpp)
# Exclude atari and chess when ALE is not available
list(FILTER SRCS EXCLUDE REGEX ".*/atari/.*")
list(FILTER SRCS EXCLUDE REGEX ".*/chess/.*")
```

**Why:**
- Atari code requires ALE headers
- Chess code has C++ standard library compatibility issues
- Not needed for Go game type

---

## ⚙️ Compilation Issues

### Issue: Missing ALE Headers

**Error:**
```
fatal error: ale_interface.hpp: No such file or directory
```

**Solution:** Excluded atari directory from compilation (see above)

### Issue: Chess Bitboard Compilation Errors

**Error:**
```
error: 'std::uint64_t' has not been declared
```

**Solution:** Excluded chess directory from compilation (see above)

**Why:** Chess implementation has missing `#include <cstdint>` causing type errors

---

## 🔗 Linking Problems

### Major Issue: CUDA Runtime Linking

**Error:**
```
undefined reference to `cudaGetDriverEntryPointByVersion@libcudart.so.12'
```

**Root Cause:** PyTorch's bundled CUDA runtime requires specific linking

**Solution Attempts:**

1. ❌ **System CUDA runtime** - Symbol not available
   ```cmake
   target_link_libraries(minizero /usr/lib/x86_64-linux-gnu/libcudart.so)
   ```

2. ❌ **Link directory** - Not found at link time
   ```cmake
   target_link_directories(minizero PRIVATE "${Python3_SITELIB}/nvidia/cuda_runtime/lib")
   ```

3. ✅ **Direct path to PyTorch's CUDA runtime** - **SUCCESS!**
   ```cmake
   target_link_libraries(
       minizero
       ${TORCH_LIBRARIES}
       /home/cthsu/miniconda3/envs/ml_hw2/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
   )
   ```

**Applied to Files:**
- `Q5/minizero/minizero/CMakeLists.txt`
- `Q5/style_detection/CMakeLists.txt`

**Why This Works:** PyTorch bundles its own CUDA runtime with the specific symbols it needs

---

## ✅ Final Build Success

### Build Command:
```bash
cd Q5
bash scripts/build.sh go
```

### Build Output:
```
[100%] Built target minizero
[100%] Built target style
```

### Executables Created:
```
build/go/minizero_go (31MB)
build/go/style_go (38MB)
build/go/minizero_py.cpython-310-x86_64-linux-gnu.so
build/go/style_py.cpython-310-x86_64-linux-gnu.so
```

---

## 🧪 Testing the Built Module

### Test Suite Created

**File:** `test_style_py.py`

**Tests:**
1. ✅ Module import and configuration loading
2. ✅ Module attributes enumeration (41 attributes found)
3. ✅ Go Environment (Env) functionality
4. ✅ Player enum values
5. ✅ DataLoader creation and methods
6. ✅ Configuration getter functions
7. ⚠️ Feature extraction (requires actual data loading)

### Example Usage Created

**File:** `example_usage.py`

**Demonstrates:**
1. ✅ Proper module import method
2. ✅ Configuration loading and access
3. ✅ Go environment simulation
4. ✅ Legal action querying
5. ✅ Move execution
6. ✅ DataLoader usage with SGF files
7. ✅ All configuration getters

### Documentation Created

**File:** `USAGE_GUIDE.md`

**Contains:**
- Complete API reference
- Import instructions
- Configuration guide
- Code examples
- Troubleshooting tips

---

## 📊 Final Results

### Successfully Built Components:

| Component | Status | Description |
|-----------|--------|-------------|
| `minizero_go` | ✅ | Main Go game executable |
| `style_go` | ✅ | Style detection executable |
| `minizero_py` | ✅ | Python bindings for minizero |
| `style_py` | ✅ | Python bindings for style detection |

### Available Python Classes:

| Class | Purpose |
|-------|---------|
| `Env` | Go game environment (12 methods) |
| `DataLoader` | SGF file loading (6 methods) |
| `SLDataLoader` | Supervised learning data |
| `Action` | Go move representation |
| `Player` | Player enum (player_1, player_2, etc.) |
| `EnvLoder` | Environment loader |

### Available Configuration Functions:

- 30+ configuration getters including:
  - `get_game_name()` → "go_19x19"
  - `get_batch_size()` → 1024
  - `get_learning_rate()` → 0.01
  - `get_nn_num_blocks()` → 1
  - `get_nn_num_hidden_channels()` → 256
  - And many more...

---

## 🎯 Key Lessons Learned

### 1. Conda Environment Benefits
Using a dedicated conda environment:
- ✅ Isolates dependencies
- ✅ Provides consistent PyTorch with CUDA
- ✅ Easy to reproduce on other systems

### 2. PyTorch CUDA Runtime
PyTorch bundles its own CUDA runtime that must be linked explicitly:
- Don't rely on system CUDA libraries
- Use PyTorch's bundled version
- Link with full path for reliability

### 3. Game Type Specific Building
When building for specific game types:
- ✅ Exclude unused game implementations
- ✅ Comment out unnecessary dependencies
- ✅ Reduces build complexity and time

### 4. pybind11 Version Matters
Always check compatibility between:
- pybind11 version
- CMake version
- Python version
- Compiler version

### 5. CMake Configuration Order
Proper order matters:
1. Find Python3 first
2. Then find pybind11
3. Then create pybind11 modules

---

## 🚀 Next Steps

Now that the module is built and tested:

1. **Load actual Go game data** (SGF files)
2. **Train neural networks** using the DataLoader
3. **Implement style detection algorithms**
4. **Run experiments** with different configurations
5. **Analyze results** using the built tools

---

## 📝 Quick Reference

### Import the module:
```python
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py
style_py.load_config_file('conf.cfg')
```

### Create Go environment:
```python
env = style_py.Env()
env.reset()
```

### Load SGF data:
```python
loader = style_py.DataLoader('/path/to/sgf/directory')
```

### Run tests:
```bash
python test_style_py.py
python example_usage.py
```

---

## 📞 Support Resources

- **Build script:** `scripts/build.sh`
- **Configuration:** `conf.cfg`
- **C++ source:** `style_detection/`, `minizero/`
- **Documentation:** `USAGE_GUIDE.md`
- **Test scripts:** `test_style_py.py`, `example_usage.py`

---

**Status:** ✅ Build environment fully configured and tested  
**Date:** October 26, 2025  
**Environment:** ml_hw2 (conda)  
**Build Type:** Release (Go game)
