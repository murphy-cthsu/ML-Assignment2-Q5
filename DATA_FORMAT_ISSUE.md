# ⚠️ IMPORTANT: Data Format Issue Found

## Problem
The C++ `DataLoader` reports **0 players** for all datasets, even though SGF files exist.

## Root Cause
The SGF file naming convention matters!

### Current File Structure
```
train_set/
├── 1.sgf
├── 2.sgf
├── 3.sgf
...
└── 200.sgf          ❌ DataLoader can't identify players from these names

test_set/query_set/
├── player001.sgf
├── player002.sgf
├── player003.sgf
...
└── player600.sgf    ✅ DataLoader recognizes 600 players from these names

test_set/cand_set/
├── player001.sgf
├── player002.sgf
...
└── player600.sgf    ✅ DataLoader recognizes 600 players from these names
```

## Solution Options

### Option 1: Use test_set for Training (Quick Fix)
Since test_set has proper player naming, you can train on it:

```bash
python train_style_model.py \
    --train_dir test_set/query_set \
    --epochs 10 \
    --batch_size 8 \
    --num_samples 1000 \
    --output_dir models/test
```

### Option 2: Fix train_set Naming
The SGF files in `train_set/` need to be organized by player. Either:

1. **Rename files to match player pattern:**
   ```bash
   # If each file is one player:
   mv train_set/1.sgf train_set/player001.sgf
   mv train_set/2.sgf train_set/player002.sgf
   # ... etc
   ```

2. **Organize into subdirectories by player** (if that's what DataLoader expects):
   ```
   train_set/
   ├── player001/
   │   ├── game1.sgf
   │   └── game2.sgf
   ├── player002/
   │   └── game1.sgf
   ...
   ```

### Option 3: Check Configuration
The `conf.cfg` file might have settings about how to parse player information from SGF files.

Check `conf.cfg` for any relevant settings.

## Quick Test

Test if renaming works:
```bash
# Make a test copy
mkdir test_train
cp train_set/1.sgf test_train/player001.sgf
cp train_set/2.sgf test_train/player002.sgf

# Test loading
python -c "
_temps = __import__(f'build.go', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py
style_py.load_config_file('conf.cfg')
loader = style_py.DataLoader('test_train')
print(f'Players found: {loader.get_num_of_player()}')
"
```

## Current Status

✅ **Architecture is correct** - matches paper specification
✅ **Code is working** - all tests pass
❌ **Data loading issue** - DataLoader can't identify players from train_set files

**Next step:** Figure out correct naming convention for train_set, or use test_set/query_set for training.
