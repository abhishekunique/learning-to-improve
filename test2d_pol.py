#!/usr/bin/env python3
"""
Trajectory Improvement using Transformer Policy with Noise-Augmented Demonstrations for Meta-World

This module implements:
1. Generate optimal demonstrations from Meta-World reaching environment
2. Add increasing noise to create progressively worse demonstrations
3. Train a Transformer-based policy to improve trajectories
4. Implement iterated inference for trajectory improvement
"""

import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
from pointmass_2d import PointMass2DEnv, generate_optimal_demonstrations, visualize_demonstration
from tensorboardX import SummaryWriter
from datetime import datetime
from expert_data_gen import get_expert_trajectory, execute_policy, obs_processor
import metaworld
import gymnasium as gym
import math
import copy


def extract_step_features(step: Dict[str, Any]) -> np.ndarray:
    """
    Extract features from a single trajectory step.
    
    Args:
        step: Dictionary containing trajectory step data with 'obs' and 'action' keys
    
    Returns:
        Feature array: [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z, action_x, action_y, action_z, action_gripper, l2_distance_to_goal]
    """
    pos = step['obs'][:3]
    goal = step['obs'][3:6]
    l2_distance = np.linalg.norm(pos - goal)
    
    features = np.concatenate([
        step['obs'][:6],  # pos and goal (6 features)
        step['action'],    # action (4 features including gripper)
        [l2_distance]      # L2 distance to goal (1 feature)
    ]).astype(np.float32)
    
    return features


def extract_trajectory_features(trajectory: List[Dict]) -> List[np.ndarray]:
    """
    Extract features from an entire trajectory.
    
    Args:
        trajectory: List of trajectory step dictionaries
    
    Returns:
        List of feature arrays for each step
    """
    return [extract_step_features(step) for step in trajectory]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sinusoidal positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and transpose
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer since it's not a parameter
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TrajectoryDataset(Dataset):
    """Dataset for trajectory improvement training with context-based action prediction."""
    
    def __init__(self, bad_trajectories: List[List[Dict]], good_trajectories: List[List[Dict]], 
                 context_length: int = 100, num_samples_per_traj: int = 5):
        """
        Args:
            bad_trajectories: List of noisy/bad trajectory demonstrations
            good_trajectories: List of corresponding optimal trajectory demonstrations
            context_length: Number of steps to use as context from bad trajectory
            num_samples_per_traj: Number of samples to generate per trajectory pair
        """
        self.bad_trajectories = []
        self.good_states = []
        self.good_actions = []
        self.context_length = context_length
        
        # Process each trajectory pair
        for bad_traj, good_traj in zip(bad_trajectories, good_trajectories):
            # Generate multiple samples per trajectory pair
            for _ in range(num_samples_per_traj):
                # Extract context from bad trajectory
                context_steps = bad_traj[:]
                context_features = []
                
                for step in context_steps:
                    features = extract_step_features(step)
                    context_features.append(features)
                
                # Sample a target state and action from good trajectory (after context start)
                target_start_idx = np.random.randint(0, len(good_traj))
                target_step = good_traj[target_start_idx]
                
                # Extract state and action separately
                target_state = target_step['obs'][:6]  # pos_x, pos_y, pos_z, goal_x, goal_y, goal_z
                target_action = target_step['action']  # action_x, action_y, action_z, action_gripper
                
                # Store context, state, and action
                self.bad_trajectories.append(np.array(context_features))  # Shape: (context_length, 11)
                self.good_states.append(target_state)  # Shape: (6,)
                self.good_actions.append(target_action)  # Shape: (4,)
    
    def __len__(self):
        return len(self.bad_trajectories)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.bad_trajectories[idx]),  # Shape: (context_length, 11)
            torch.FloatTensor(self.good_states[idx]),        # Shape: (6,)
            torch.FloatTensor(self.good_actions[idx])        # Shape: (4,)
        ) 


class TrajectoryImproverTransformer(nn.Module):
    """Transformer-based policy model for context-based action prediction."""
    
    def __init__(self, 
                 feature_dim: int = 11,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 state_dim: int = 6,
                 action_dim: int = 4,
                 hidden_dim: int = 256,
                 num_mlp_layers: int = 3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Transformer output projection
        self.transformer_output_projection = nn.Linear(d_model, hidden_dim)
        
        # MLP layers for action prediction
        # Input: transformer_output + state
        mlp_input_dim = hidden_dim + state_dim
        
        layers = []
        layer_sizes = [mlp_input_dim] + [hidden_dim] * num_mlp_layers + [action_dim]
        
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:  # Don't add activation after last layer
                layers.append(nn.ReLU())
        
        self.action_mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, context, state):
        """
        Predict action given context trajectory and current state.
        
        Args:
            context: Context trajectory features, shape: (batch_size, seq_len, feature_dim)
            state: Current state, shape: (batch_size, state_dim)
        
        Returns:
            Predicted action: Shape: (batch_size, action_dim)
        """
        # Project context to transformer dimension
        context = self.input_projection(context)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        context = context.transpose(0, 1)  # (seq_len, batch_size, d_model)
        context = self.pos_encoder(context)
        context = context.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(context)  # (batch_size, seq_len, d_model)
        
        # Global average pooling over sequence dimension
        transformer_output = torch.mean(transformer_output, dim=1)  # (batch_size, d_model)
        
        # Project transformer output
        transformer_output = self.transformer_output_projection(transformer_output)  # (batch_size, hidden_dim)
        
        # Concatenate transformer output with state
        combined = torch.cat([transformer_output, state], dim=1)  # (batch_size, hidden_dim + state_dim)
        
        # Predict action through MLP
        action = self.action_mlp(combined)  # (batch_size, action_dim)
        
        return action 


def add_noise_to_trajectory(env, env_name, trajectory: List[Dict], noise_level: float, max_episode_length: int = 100) -> List[Dict]:
    """
    Add noise to a trajectory to create a worse version.
    
    Args:
        trajectory: List of trajectory steps
        noise_level: Standard deviation of noise to add
    
    Returns:
        Noisy trajectory
    """
    # Extract goal from first step of trajectory
    original_goal = copy.deepcopy(trajectory[0]['obs'][3:6])
    
    # Add noise to goal
    goal_noise = np.random.uniform(-noise_level, noise_level, 3)
    noisy_goal = original_goal + goal_noise

    noisy_trajectory = get_expert_trajectory(env, env_name, goal_pos=noisy_goal, max_traj_len=max_episode_length, overwrite_goal=original_goal)

    return noisy_trajectory


def generate_noise_augmented_data(env, env_name: str,
                                 num_optimal_demos: int = 50,
                                 noise_levels: List[float] = None,
                                 max_episode_length: int = 100) -> Tuple[List, List]:
    """
    Generate optimal demonstrations and create noisy versions for training.
    
    Args:
        env: The environment
        env_name: Environment name
        num_optimal_demos: Number of optimal demonstrations to generate
        noise_levels: List of noise levels to apply
    
    Returns:
        Tuple of (bad_trajectories, good_trajectories)
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"Generating {num_optimal_demos} optimal demonstrations...")
    low_val = np.array([-0.09440674, 0.8])
    high_val = np.array([0.09411726, 0.9])
    optimal_demos = []
    
    # Generating goals in the plane
    for _ in range(num_optimal_demos):
        goal_pos = np.concatenate([np.random.uniform(low_val, high_val), np.array([0.14507392])])
        traj = get_expert_trajectory(env, env_name, max_traj_len=max_episode_length, goal_pos=goal_pos)
        optimal_demos.append(traj)
    
    bad_trajectories = []
    good_trajectories = []
    
    print("Creating noisy versions...")
    num_trajectories = 0
    for demo_idx, demo in enumerate(optimal_demos):
        print(f"Processing demonstration {demo_idx+1}/{len(optimal_demos)}")
        # Create noisy versions for each noise level
        for i, noise_level in enumerate(noise_levels):
            num_trajectories += 1
            total_trajectories = len(optimal_demos) * len(noise_levels)
            print(f"  Created trajectory {num_trajectories}/{total_trajectories} with noise level {noise_level}")
            # Bad trajectory with current noise level
            bad_demo = add_noise_to_trajectory(env, env_name, demo, noise_level)
            bad_trajectories.append(bad_demo)
            
            # Good trajectory with one level less noise (or optimal if at first level)
            if i == 0:
                # For the first noise level, use the optimal demo as the good trajectory
                good_trajectories.append(demo)
            else:
                # Use the previous noise level as the good trajectory
                good_demo = add_noise_to_trajectory(env, env_name, demo, noise_levels[i-1])
                good_trajectories.append(good_demo)
        
        # Add final pair where both good and bad are noise-free (end of improvement chain)
        bad_trajectories.append(demo)  # Noise-free bad trajectory
        good_trajectories.append(demo)  # Noise-free good trajectory (same as bad)
    
    print(f"Generated {len(bad_trajectories)} bad trajectories and {len(good_trajectories)} good trajectories")
    
    print("trajectory viz")
    
    plt.cla()
    plt.clf()
    for traj in bad_trajectories:
        obs_list = np.array([ts['obs'] for ts in traj])
        plt.plot(obs_list[:, 0], obs_list[:, 1])
        plt.scatter(obs_list[-1, -3], obs_list[-1, -2], marker='x')
    plt.savefig('test_data_bad.png')

    plt.cla()
    plt.clf()
    for traj in good_trajectories:
        obs_list = np.array([ts['obs'] for ts in traj])
        plt.plot(obs_list[:, 0], obs_list[:, 1])
        plt.scatter(obs_list[-1, -3], obs_list[-1, -2], marker='x')
    plt.savefig('test_data_good.png')
    
    return bad_trajectories, good_trajectories 


def train_trajectory_improver(bad_trajectories: List[List[Dict]], 
                             good_trajectories: List[List[Dict]],
                             epochs: int = 100,
                             batch_size: int = 32,
                             learning_rate: float = 1e-3,
                             writer: SummaryWriter = None,
                             context_length: int = 100,
                             num_samples_per_traj: int = 5,
                             model: TrajectoryImproverTransformer = None) -> TrajectoryImproverTransformer:
    """
    Train the trajectory improvement transformer policy.
    
    Args:
        bad_trajectories: List of noisy/bad trajectory demonstrations
        good_trajectories: List of corresponding optimal trajectory demonstrations
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        writer: TensorboardX SummaryWriter for logging
        context_length: Number of steps to use as context from bad trajectory
        num_samples_per_traj: Number of samples to generate per trajectory pair
    
    Returns:
        Trained TrajectoryImproverTransformer model
    """
    # Create random indices for shuffling
    num_trajectories = len(bad_trajectories)
    indices = np.random.permutation(num_trajectories)
    
    # Split data into training and validation sets (80/20 split)
    num_train = int(0.8 * num_trajectories)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Use indices to create shuffled train/val splits
    train_bad = [bad_trajectories[i] for i in train_indices]
    train_good = [good_trajectories[i] for i in train_indices]
    val_bad = [bad_trajectories[i] for i in val_indices]
    val_good = [good_trajectories[i] for i in val_indices]
    
    # Create datasets and dataloaders with context-based sampling
    train_dataset = TrajectoryDataset(train_bad, train_good, context_length=context_length, num_samples_per_traj=num_samples_per_traj)
    val_dataset = TrajectoryDataset(val_bad, val_good, context_length=context_length, num_samples_per_traj=num_samples_per_traj)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if model is None:
        # Initialize model
        model = TrajectoryImproverTransformer(
            feature_dim=11,  # 6 obs + 4 action + 1 distance
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            state_dim=6,  # pos_x, pos_y, pos_z, goal_x, goal_y, goal_z
            action_dim=4,  # action_x, action_y, action_z, action_gripper
            hidden_dim=256,
            num_mlp_layers=3
        )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training trajectory improver transformer policy for {epochs} epochs...")
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    print(f"Context length: {context_length}, Samples per trajectory: {num_samples_per_traj}")
    
    best_val_loss = float('inf')
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        num_train_batches = 0
        
        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (batch_context, batch_states, batch_actions) in enumerate(batch_pbar):
            optimizer.zero_grad()
            
            # batch_context shape: (batch_size, seq_len, feature_dim)
            # batch_states shape: (batch_size, state_dim)
            # batch_actions shape: (batch_size, action_dim)
            
            predicted_actions = model(batch_context, batch_states)
            loss = criterion(predicted_actions, batch_actions)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
            
            batch_pbar.set_postfix({'Train Loss': f'{loss.item():.6f}'})
            
            if writer:
                writer.add_scalar('Loss/train_batch', loss.item(), 
                                epoch * len(train_dataloader) + batch_idx)
        
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_context, batch_states, batch_actions in val_dataloader:
                predicted_actions = model(batch_context, batch_states)
                loss = criterion(predicted_actions, batch_actions)
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.6f}',
            'Val Loss': f'{avg_val_loss:.6f}'
        })
        
        # Log metrics to tensorboard
        if writer:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Log model parameter histograms
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', 
                                   param.data.cpu().numpy(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}',
                                       param.grad.cpu().numpy(), epoch)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if writer:
                writer.add_scalar('Best_validation_loss', best_val_loss, epoch)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    return model 


def improve_trajectory(model: TrajectoryImproverTransformer, 
                      trajectory: List[Dict],
                      context_length: int = 100,
                      env = None) -> List[Dict]:
    """
    Improve a single trajectory using the trained transformer policy with closed-loop execution.
    
    Args:
        model: Trained trajectory improvement model
        trajectory: Input trajectory to improve
        context_length: Number of steps to use as context
        env: Environment to step through
    
    Returns:
        Improved trajectory
    """
    # model.eval()
    improved_trajectory = []
    
    # Reset environment to initial state
    obs, info = env.reset()
    goal = trajectory[0]['obs'][-3:].copy()

    # Setting the environment goal
    env.env.env.env.env.env.env.env._target_pos = goal.copy()
    obs = env.env.env.env.env.env.env.env._get_obs()
    
    
    with torch.no_grad():
        # Extract features for context trajectory
        features_list = extract_trajectory_features(trajectory)
        context_tensor = torch.FloatTensor(np.stack(features_list)).unsqueeze(0)  # Shape: (1, seq_len, 11)
        
        # Run policy with fixed context
        for i in range(len(trajectory)):
            # Get current state
            current_state = obs_processor(obs)  # pos_x, pos_y, pos_z, goal_x, goal_y, goal_z
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)  # Shape: (1, 6)
            
            # Get model prediction for this step
            predicted_action = model(context_tensor, state_tensor).squeeze(0).numpy()  # Shape: (4,)
            
            # Step environment with predicted action
            next_obs, reward, done, truncated, info = env.step(predicted_action)
            
            # Create improved step
            improved_step = {
                'obs': obs_processor(copy.deepcopy(obs)),
                'action': copy.deepcopy(predicted_action),
                'reward': reward,
                'done': done,
                'info': info
            }
            improved_trajectory.append(improved_step)
            obs = next_obs.copy()
    
    return improved_trajectory


def iterated_inference(model: TrajectoryImproverTransformer, 
                      trajectory: List[Dict], 
                      num_iterations: int = 5,  
                      convergence_threshold: float = 1e-4,
                      env = None) -> Tuple[List[Dict], List[List[Dict]]]:
    """
    Apply iterated inference to progressively improve a trajectory.
    
    Args:
        model: Trained trajectory improvement model
        trajectory: Input trajectory to improve
        num_iterations: Maximum number of improvement iterations
        convergence_threshold: Threshold for convergence detection
        env: Environment for stepping
    
    Returns:
        Tuple of (final_improved_trajectory, list_of_all_trajectories)
    """
    trajectory_list = [copy.deepcopy(trajectory)]
    current_trajectory = copy.deepcopy(trajectory)
    previous_trajectory = None
    
    print(f"Starting iterated inference with {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        # Improve trajectory
        improved_trajectory = improve_trajectory(model, current_trajectory, context_length=100, env=env)
        
        # Check convergence
        if previous_trajectory is not None:
            # Calculate average change in positions
            total_change = 0
            for prev_step, curr_step in zip(previous_trajectory, improved_trajectory):
                pos_change = np.linalg.norm(
                    prev_step['obs'][:3] - curr_step['obs'][:3]
                )
                total_change += pos_change
            
            avg_change = total_change / len(improved_trajectory)
            print(f"Iteration {iteration + 1}: Average position change = {avg_change:.9f}")
            
            # if avg_change < convergence_threshold:
            #     print(f"Converged after {iteration + 1} iterations!")
            #     break
        
        previous_trajectory = copy.deepcopy(current_trajectory)
        current_trajectory = copy.deepcopy(improved_trajectory)
        trajectory_list.append(current_trajectory.copy())
    # import pdb; pdb.set_trace()
    return current_trajectory, trajectory_list


def evaluate_trajectory_quality(trajectory: List[Dict], env) -> Dict[str, float]:
    """
    Evaluate the quality of a trajectory.
    
    Args:
        trajectory: Trajectory to evaluate
        env: Environment for reference
    
    Returns:
        Dictionary with quality metrics
    """
    if not trajectory:
        return {'total_reward': 0, 'success': False, 'final_distance': float('inf')}
    
    # Compute metrics from observations
    total_reward = 0
    for step in trajectory:
        goal_pos = step['obs'][3:6]  # Goal position is stored in obs[3:6]
        agent_pos = step['obs'][:3]  # Agent position is stored in obs[:3]
        distance = np.linalg.norm(agent_pos - goal_pos)
        reward = -distance  # Reward is negative distance to goal
        total_reward += reward
    
    final_distance = np.linalg.norm(trajectory[-1]['obs'][:3] - trajectory[-1]['obs'][3:6])
    success = final_distance < 0.5  # Assuming success threshold of 0.5
    
    return {
        'total_reward': total_reward,
        'success': success,
        'final_distance': final_distance,
        'num_steps': len(trajectory)
    } 


def visualize_trajectory_comparison(env, 
                                  original_trajectory: List[Dict],
                                  improved_trajectory: List[Dict],
                                  trajectory_list: List[List[Dict]],
                                  overall_mins: np.ndarray,
                                  overall_maxs: np.ndarray,
                                  save_path: str = None):
    """
    Visualize comparison between original and improved trajectories in 2D.
    
    Args:
        env: The environment
        original_trajectory: Original trajectory
        improved_trajectory: Improved trajectory
        trajectory_list: List of intermediate trajectories
        overall_mins: Minimum bounds for 2D plot
        overall_maxs: Maximum bounds for 2D plot
        save_path: Optional path to save visualization
    """
    
    fig = plt.figure(figsize=(20, 8))
    
    # Create 2D subplots
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # Plot original trajectory in 2D
    ax1.set_title("Original Trajectory (2D)", fontsize=14)
    ax1.set_xlim(overall_mins[0], overall_maxs[0])
    ax1.set_ylim(overall_mins[1], overall_maxs[1])
    ax1.grid(True, alpha=0.3)
    
    # Extract 2D positions (x, y) - dropping z dimension
    positions_2d = []
    for i, step in enumerate(original_trajectory):
        pos_2d = step['obs'][:2]  # Get x, y position only
        positions_2d.append([pos_2d[0], pos_2d[1]])
    
    positions_2d = np.array(positions_2d)
    
    # Plot 2D trajectory
    ax1.plot(positions_2d[:, 0], positions_2d[:, 1], 
             'b-', alpha=0.8, linewidth=3, label='Trajectory')
    
    # Plot start and end points
    ax1.scatter(positions_2d[0, 0], positions_2d[0, 1], 
                c='green', s=200, label='Start', marker='o')
    ax1.scatter(positions_2d[-1, 0], positions_2d[-1, 1], 
                c='red', s=200, label='End', marker='o')
    
    # Plot goal in 2D
    goal = original_trajectory[0]['obs'][3:5]  # Only x, y coordinates of goal
    ax1.scatter(goal[0], goal[1], c='orange', s=200, label='Goal', marker='*')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()
    
    # Plot improved trajectory and intermediates in 2D
    ax2.set_title("Improvement Process (2D)", fontsize=14)
    ax2.set_xlim(overall_mins[0], overall_maxs[0])
    ax2.set_ylim(overall_mins[1], overall_maxs[1])
    ax2.grid(True, alpha=0.3)
    
    # Create color map for trajectory evolution
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_list)))
    
    # Plot intermediate trajectories with color gradient
    for i, (traj, color) in enumerate(zip(trajectory_list, colors)):
        positions_2d = []
        for j, step in enumerate(traj):
            pos_2d = step['obs'][:2]  # Only x, y coordinates
            positions_2d.append([pos_2d[0], pos_2d[1]])
        
        positions_2d = np.array(positions_2d)
        label = f'Iteration {i}'
        ax2.plot(positions_2d[:, 0], positions_2d[:, 1], 
                '-', color=color, alpha=0.6, linewidth=2, label=label)
    
    # Plot start and end points of final improved trajectory
    final_positions_2d = []
    for i, step in enumerate(improved_trajectory):
        pos_2d = step['obs'][:2]  # Only x, y coordinates
        final_positions_2d.append([pos_2d[0], pos_2d[1]])
    
    final_positions_2d = np.array(final_positions_2d)
    ax2.scatter(final_positions_2d[0, 0], final_positions_2d[0, 1], 
                c='green', s=200, label='Start', marker='o')
    ax2.scatter(final_positions_2d[-1, 0], final_positions_2d[-1, 1], 
                c='red', s=200, label='End', marker='o')
    
    # Plot goal in 2D
    goal = improved_trajectory[0]['obs'][3:5]  # Only x, y coordinates of goal
    ax2.scatter(goal[0], goal[1], c='orange', s=200, label='Goal', marker='*')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_training_data(env,
                          bad_trajectories: List[List[Dict]], 
                          good_trajectories: List[List[Dict]], 
                          overall_mins: np.ndarray,
                          overall_maxs: np.ndarray,
                          num_samples: int = 5,
                          save_path: str = None):
    """
    Visualize a sample of training data showing bad and good trajectory pairs in 2D.
    
    Args:
        env: The environment
        bad_trajectories: List of noisy/bad trajectories
        good_trajectories: List of corresponding optimal trajectories 
        overall_mins: Minimum bounds for 2D plot
        overall_maxs: Maximum bounds for 2D plot
        num_samples: Number of trajectory pairs to visualize
        save_path: Optional path to save the visualization
    """
    
    # Create figure with 2D subplots
    fig = plt.figure(figsize=(20, 4*num_samples))
    fig.suptitle("Training Data: Bad vs Good Trajectories (2D)", fontsize=16)
    
    # Randomly sample trajectory pairs
    indices = np.random.choice(len(bad_trajectories), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        bad_traj = bad_trajectories[idx]
        good_traj = good_trajectories[idx]
        
        # Create 2D subplots for this pair
        ax1 = fig.add_subplot(num_samples, 2, 2*i + 1)
        ax2 = fig.add_subplot(num_samples, 2, 2*i + 2)
        
        # Plot bad trajectory in 2D
        ax1.set_title(f"Bad Trajectory {i+1} (2D)", fontsize=12)
        ax1.set_xlim(overall_mins[0], overall_maxs[0])
        ax1.set_ylim(overall_mins[1], overall_maxs[1])
        ax1.grid(True, alpha=0.3)
        
        # Extract 2D positions for bad trajectory
        bad_positions_2d = []
        for step in bad_traj:
            pos_2d = step['obs'][:2]  # Only x, y coordinates
            bad_positions_2d.append(pos_2d)
        
        bad_positions_2d = np.array(bad_positions_2d)
        
        # Plot 2D trajectory
        ax1.plot(bad_positions_2d[:, 0], bad_positions_2d[:, 1], 
                 'r-', alpha=0.8, linewidth=3, label='Bad Trajectory')
        
        # Plot start and end points
        ax1.scatter(bad_positions_2d[0, 0], bad_positions_2d[0, 1], 
                    c='green', s=150, label='Start', marker='o')
        ax1.scatter(bad_positions_2d[-1, 0], bad_positions_2d[-1, 1], 
                    c='red', s=150, label='End', marker='o')
        
        # Plot goal in 2D
        goal = bad_traj[0]['obs'][3:5]  # Only x, y coordinates of goal
        ax1.scatter(goal[0], goal[1], 
                   c='orange', s=150, label='Goal', marker='*')
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        
        # Plot good trajectory in 2D
        ax2.set_title(f"Good Trajectory {i+1} (2D)", fontsize=12)
        ax2.set_xlim(overall_mins[0], overall_maxs[0])
        ax2.set_ylim(overall_mins[1], overall_maxs[1])
        ax2.grid(True, alpha=0.3)
        
        # Extract 2D positions for good trajectory
        good_positions_2d = []
        for step in good_traj:
            pos_2d = step['obs'][:2]  # Only x, y coordinates
            good_positions_2d.append(pos_2d)
        
        good_positions_2d = np.array(good_positions_2d)
        
        # Plot 2D trajectory
        ax2.plot(good_positions_2d[:, 0], good_positions_2d[:, 1], 
                 'b-', alpha=0.8, linewidth=3, label='Good Trajectory')
        
        # Plot start and end points
        ax2.scatter(good_positions_2d[0, 0], good_positions_2d[0, 1], 
                    c='green', s=150, label='Start', marker='o')
        ax2.scatter(good_positions_2d[-1, 0], good_positions_2d[-1, 1], 
                    c='red', s=150, label='End', marker='o')
        
        # Plot goal in 2D
        goal = good_traj[0]['obs'][3:5]  # Only x, y coordinates of goal
        ax2.scatter(goal[0], goal[1],
                   c='orange', s=150, label='Goal', marker='*')
        
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.legend()
    
    plt.tight_layout()

    plt.savefig('training_data_viz.png') 


def main():
    """Main function to demonstrate transformer-based trajectory improvement for Meta-World."""
    
    # Create environment
    print("Creating Reaching Environment...")
    env_name = "reach-v3"
    max_traj_len = 100
    env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array")
    
    # Generate training data
    print("\n=== Generating Training Data ===")
    bad_trajectories, good_trajectories = generate_noise_augmented_data(
        env, env_name, num_optimal_demos=1000, noise_levels=[0.2, 0.4, 0.6, 0.8, 1.0], max_episode_length=max_traj_len
    )
    assert len(bad_trajectories) == len(good_trajectories), "Bad and good trajectory lists must have same length"
    for bad_traj, good_traj in zip(bad_trajectories, good_trajectories):
        assert len(bad_traj) == len(good_traj), "Each corresponding bad and good trajectory must have same length"
    
    # Find min/max ranges per dimension for trajectory positions
    good_pos_mins = np.array([float('inf')] * 3)  # x,y,z minimums
    good_pos_maxs = np.array([float('-inf')] * 3)  # x,y,z maximums
    bad_pos_mins = np.array([float('inf')] * 3)
    bad_pos_maxs = np.array([float('-inf')] * 3)

    for good_traj, bad_traj in zip(good_trajectories, bad_trajectories):
        for step in good_traj:
            pos = step['obs'][:3]
            good_pos_mins = np.minimum(good_pos_mins, pos)
            good_pos_maxs = np.maximum(good_pos_maxs, pos)
            
        for step in bad_traj:
            pos = step['obs'][:3]
            bad_pos_mins = np.minimum(bad_pos_mins, pos)
            bad_pos_maxs = np.maximum(bad_pos_maxs, pos)
    
    # Get overall min/max across both good and bad trajectories
    overall_mins = np.minimum(good_pos_mins, bad_pos_mins)
    overall_maxs = np.maximum(good_pos_maxs, bad_pos_maxs)
            
    print("\nPosition ranges per dimension (x,y,z):")
    print(f"Good trajectories - Mins: {good_pos_mins}, Maxs: {good_pos_maxs}")
    print(f"Bad trajectories  - Mins: {bad_pos_mins}, Maxs: {bad_pos_maxs}")
    print(f"Overall ranges   - Mins: {overall_mins}, Maxs: {overall_maxs}")
    visualize_training_data(env, bad_trajectories, good_trajectories, overall_mins, overall_maxs)

    # Train the trajectory improver
    print("\n=== Training Trajectory Improver Transformer Policy ===")
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(f'runs/trajectory_improver_transformer_policy_metaworld_{datetime.now().strftime("%Y%m%d")}')
    
    # Try to load existing model first
    model_path = 'models/trajectory_improver_transformer_policy_metaworld_2d.pt'
    load_model = False
    train_model = False
    if load_model and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = torch.load(model_path)
        if train_model:
            model = train_trajectory_improver(
                bad_trajectories, good_trajectories, 
                epochs=100, batch_size=128, learning_rate=1e-3,
                writer=writer, context_length=100, num_samples_per_traj=10,
                model=model
            )
        else:
            print(f"Loaded existing model from {model_path}")
    else:
        print("Training new transformer policy model...")
        model = train_trajectory_improver(
            bad_trajectories, good_trajectories, 
            epochs=100, batch_size=128, learning_rate=1e-3,
            writer=writer, context_length=100, num_samples_per_traj=10
        )
        # Save the trained model
        print(f"Saving model to {model_path}")
        torch.save(model, model_path)
    
    writer.close()

    # Test on a new noisy trajectory
    print("\n=== Testing Trajectory Improvement ===")
    num_eval_trajectories = 10
    improvements_noisy = []
    improvements_random = []
    for i in range(num_eval_trajectories):
        # Evaluate on a new noisy trajectory
        optimal_demo = get_expert_trajectory(env, env_name, max_traj_len=max_traj_len)
        noisy_demo = add_noise_to_trajectory(env, env_name, optimal_demo, noise_level=1.0)
        original_quality = evaluate_trajectory_quality(noisy_demo, env)
        print(f"Original trajectory quality: {original_quality}")
        improved_demo, trajectory_list = iterated_inference(model, noisy_demo, num_iterations=5, env=env)
        improved_quality = evaluate_trajectory_quality(improved_demo, env)
        print(f"Improved trajectory quality: {improved_quality}")
        improvements_noisy.append(improved_quality['final_distance'] - original_quality['final_distance'])
        visualize_trajectory_comparison(env, noisy_demo, improved_demo, trajectory_list, save_path=f'plots/mw_transformer_policy_trajectory_comparison_improvement_{i}.png', overall_mins=overall_mins, overall_maxs=overall_maxs)

        # Evaluate on a new random trajectory
        random_demo = get_expert_trajectory(env, env_name, max_traj_len=max_traj_len, random_actions=True)
        random_quality = evaluate_trajectory_quality(random_demo, env)
        print(f"Random trajectory quality: {random_quality}")
        improved_random, trajectory_list = iterated_inference(model, random_demo, num_iterations=5, env=env)
        improved_random_quality = evaluate_trajectory_quality(improved_random, env)
        print(f"Improved random trajectory quality: {improved_random_quality}")
        improvements_random.append(improved_random_quality['final_distance'] - random_quality['final_distance'])
        visualize_trajectory_comparison(env, random_demo, improved_random, trajectory_list, save_path=f'plots/mw_transformer_policy_trajectory_comparison_improvement_random_{i}.png', overall_mins=overall_mins, overall_maxs=overall_maxs)

    print(f"Average improvement for noisy trajectories: {np.mean(improvements_noisy)}")
    print(f"Average improvement for random trajectories: {np.mean(improvements_random)}")

if __name__ == "__main__":
    main() 