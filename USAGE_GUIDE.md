# Style_py Module Usage Guide

## Overview
The `style_py` module provides Python bindings to the C++ Go game environment for style detection, reinforcement learning, and game analysis.

## Building the Module

```bash
# Make sure you're in the ml_hw2 conda environment
conda activate ml_hw2

# Build for Go
bash scripts/build.sh go

# The module will be built in: build/go/style_py.cpython-310-x86_64-linux-gnu.so
```

## Importing the Module

```python
# Method 1: Direct import (recommended)
game_type = 'go'
_temps = __import__(f'build.{game_type}', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py

# Load configuration
style_py.load_config_file('conf.cfg')
```

## Available Classes and Functions

### 1. **Configuration Management**
```python
# Load configuration
style_py.load_config_file('conf.cfg')

# Access configuration values
game_name = style_py.get_game_name()  # 'go_19x19'
board_size = style_py.get_nn_input_channel_height()  # 19
input_channels = style_py.get_nn_num_input_channels()  # 18
action_size = style_py.get_nn_action_size()  # 362 (19x19 + 1 pass)

# Training parameters
batch_size = style_py.get_batch_size()
learning_rate = style_py.get_learning_rate()
```

### 2. **Go Environment (Env)**
Simulate and interact with Go games.

```python
# Create environment
env = style_py.Env()
env.reset()

# Get game state
legal_actions = env.get_legal_actions()  # List of legal moves
is_terminal = env.is_terminal()  # Check if game ended
current_player = env.get_turn()  # Player.player_1 or Player.player_2

# Make moves
action = legal_actions[0]
env.act(action)

# Extract features (for neural network input)
features = env.get_features()  # Board state representation
action_features = env.get_action_features()  # Action-specific features

# Get game information
eval_score = env.get_eval_score()  # Game evaluation
action_history = env.get_action_history()  # Move sequence
```

### 3. **DataLoader**
Load and process SGF files for training.

```python
# Create DataLoader with directory containing SGF files
loader = style_py.DataLoader('/path/to/sgf/directory')

# Check loaded data
num_players = loader.get_num_of_player()

# Load data from file
loader.load_data_from_file()

# Extract features and labels for training
# Arguments: player_id, game_id, move_step, use_random_rotation
features, labels = loader.get_feature_and_label(0, 0, 4, False)

# Get random features (with data augmentation)
random_features = loader.get_random_feature_and_label(0, 0, 4, True)

# Utility methods
loader.Check_Sgf()  # Validate SGF files
loader.Clear_Sgf()  # Clear loaded data
```

### 4. **Player Enum**
Represents players in the game.

```python
style_py.player_1      # Black player
style_py.player_2      # White player  
style_py.player_none   # No player (empty)
style_py.player_size   # Total number of player types
```

### 5. **Action Class**
Represents a move in Go.

```python
# Actions are returned by get_legal_actions()
actions = env.get_legal_actions()
action = actions[0]

# Use action with environment
env.act(action)
```

### 6. **SLDataLoader**
Supervised learning data loader (if available).

```python
sl_loader = style_py.SLDataLoader()
# Use for supervised learning tasks
```

## Configuration File (conf.cfg)

Key parameters you can modify:

```ini
# Environment
env_board_size=19          # Board size (9, 13, or 19)
env_go_komi=7.5           # Komi value

# Neural Network
nn_num_blocks=1           # ResNet blocks
nn_num_hidden_channels=256  # Hidden layer size

# Training
learner_batch_size=1024   # Batch size
learner_learning_rate=0.01  # Learning rate
learner_momentum=0.9      # SGD momentum

# Style Detection
players_per_batch=20      # Players per training batch
games_per_player=9        # Games per player
n_frames=10               # Number of frames
```

## Complete Example

```python
import os
import numpy as np

# Import and configure
game_type = 'go'
_temps = __import__(f'build.{game_type}', globals(), locals(), ['style_py'], 0)
style_py = _temps.style_py
style_py.load_config_file('conf.cfg')

# Example 1: Play a game
env = style_py.Env()
env.reset()

while not env.is_terminal():
    legal_actions = env.get_legal_actions()
    if not legal_actions:
        break
    
    # Make a random move
    import random
    action = random.choice(legal_actions)
    env.act(action)
    
    print(f"Player {env.get_turn()} made a move")

print("Game finished!")
print(f"Final score: {env.get_eval_score()}")

# Example 2: Load training data
loader = style_py.DataLoader('path/to/sgf/files')
print(f"Loaded {loader.get_num_of_player()} players")

# Extract training data
for player_id in range(loader.get_num_of_player()):
    features, labels = loader.get_feature_and_label(player_id, 0, 0, False)
    # Use features and labels for training
    # features: board state
    # labels: move to make (policy) and game outcome (value)
```

## Troubleshooting

### Module not found
```bash
# Make sure you've built the project
bash scripts/build.sh go

# Check that the .so file exists
ls -la build/go/style_py*.so
```

### std::bad_alloc error
This usually means the DataLoader needs actual data loaded:
```python
loader = style_py.DataLoader('path/to/sgf/files')
loader.load_data_from_file()  # Load the data first
```

### Configuration not found
```bash
# Make sure conf.cfg exists
ls -la conf.cfg

# Load it explicitly
style_py.load_config_file('conf.cfg')
```

## Testing

Run the provided test scripts:

```bash
# Basic functionality test
python test_style_py.py

# Practical usage examples
python example_usage.py
```

## For More Information

- Check the C++ source: `style_detection/pybind.cpp`
- Data loader implementation: `style_detection/sd_data_loader.cpp`
- Go environment: `minizero/minizero/environment/go/`
