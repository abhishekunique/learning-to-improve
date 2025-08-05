#!/usr/bin/env python3
"""
Trajectory Improvement using RNN with Noise-Augmented Demonstrations

This module implements:
1. Generate optimal demonstrations from 2D pointmass environment
2. Add increasing noise to create progressively worse demonstrations
3. Train an RNN to improve trajectories
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


def extract_step_features(step: Dict[str, Any]) -> np.ndarray:
    """
    Extract features from a single trajectory step.
    
    Args:
        step: Dictionary containing trajectory step data with 'obs' and 'action' keys
    
    Returns:
        Feature array: [pos_x, pos_y, goal_x, goal_y, action_x, action_y, l2_distance_to_goal]
    """
    pos = step['obs'][:2]
    goal = step['obs'][2:4]
    l2_distance = np.linalg.norm(pos - goal)
    
    features = np.concatenate([
        step['obs'][:4],  # pos and goal
        step['action'],
        [l2_distance]  # L2 distance to goal
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


class TrajectoryDataset(Dataset):
    """Dataset for trajectory improvement training with RNN."""
    
    def __init__(self, bad_trajectories: List[List[Dict]], good_trajectories: List[List[Dict]], max_length: int = 20):
        """
        Args:
            bad_trajectories: List of noisy/bad trajectory demonstrations
            good_trajectories: List of corresponding optimal trajectory demonstrations
            max_length: Maximum trajectory length for padding
        """
        self.bad_trajectories = []
        self.good_trajectories = []
        self.max_length = max_length
        
        # Process each trajectory pair
        for bad_traj, good_traj in zip(bad_trajectories, good_trajectories):
            # Pad trajectories to max_length
            bad_padded = self._pad_trajectory(bad_traj, max_length)
            good_padded = self._pad_trajectory(good_traj, max_length)
            
            # Extract features for each step
            bad_features = []
            good_features = []
            
            for i in range(max_length):
                bad_step = bad_padded[i]
                good_step = good_padded[i]
                
                # Extract features using the utility function
                bad_step_features = extract_step_features(bad_step)
                good_step_features = extract_step_features(good_step)
                
                bad_features.append(bad_step_features)
                good_features.append(good_step_features)
            
            # Convert to tensors
            bad_tensor = torch.FloatTensor(bad_features)  # Shape: (max_length, 7)
            good_tensor = torch.FloatTensor(good_features)  # Shape: (max_length, 7)
            
            self.bad_trajectories.append(bad_tensor)
            self.good_trajectories.append(good_tensor)
    
    def _pad_trajectory(self, trajectory: List[Dict], max_length: int) -> List[Dict]:
        """Pad trajectory to max_length by repeating the last step."""
        if len(trajectory) >= max_length:
            return trajectory[:max_length]
        
        padded = trajectory.copy()
        last_step = trajectory[-1] if trajectory else None
        
        while len(padded) < max_length:
            if last_step:
                padded.append(last_step.copy())
            else:
                # Create a dummy step if trajectory is empty
                dummy_step = {
                    'obs': np.zeros(4),
                    'action': np.zeros(2),
                    'reward': 0.0,
                    'done': True,
                    'info': {'distance_to_goal': float('inf'), 'success': False}
                }
                padded.append(dummy_step)
        
        return padded
    
    def __len__(self):
        return len(self.bad_trajectories)
    
    def __getitem__(self, idx):
        return (
            self.bad_trajectories[idx],  # Shape: (max_length, 7)
            self.good_trajectories[idx]   # Shape: (max_length, 7)
        )


class TrajectoryImproverRNN(nn.Module):
    """RNN model for improving trajectories."""
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, num_layers: int = 2, 
                 rnn_type: str = 'lstm', dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'lstm' or 'gru'")
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        if self.rnn_type == 'lstm':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        else:  # GRU
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward pass through RNN
        rnn_out, _ = self.rnn(x, hidden)
        
        # Project to output dimension
        output = self.output_projection(rnn_out)
        
        return output  # Shape: (batch_size, seq_len, input_dim)


def add_noise_to_trajectory(trajectory: List[Dict], noise_level: float, max_episode_length: int = 20) -> List[Dict]:
    """
    Add noise to a trajectory to create a worse version.
    
    Args:
        trajectory: List of trajectory steps
        noise_level: Standard deviation of noise to add
    
    Returns:
        Noisy trajectory
    """
    # Extract goal from first step of trajectory
    original_goal = trajectory[0]['obs'][2:4]
    
    # Add noise to goal
    goal_noise = np.random.uniform(-noise_level, noise_level, 2)
    noisy_goal = original_goal + goal_noise
    
    # Create environment and set noisy goal
    env = PointMass2DEnv()
    env.goal = noisy_goal
    
    # Generate optimal trajectory to noisy goal
    noisy_trajectory = generate_optimal_demonstrations(env, target=noisy_goal, num_demos=1, max_episode_length=max_episode_length)[0]
    
    # Replace noisy goal with original goal in observations
    for step in noisy_trajectory:
        step['obs'][2:4] = original_goal
    
    return noisy_trajectory


def generate_noise_augmented_data(env: PointMass2DEnv, 
                                 num_optimal_demos: int = 50,
                                 noise_levels: List[float] = None) -> Tuple[List, List]:
    """
    Generate optimal demonstrations and create noisy versions for training.
    
    Args:
        env: The pointmass environment
        num_optimal_demos: Number of optimal demonstrations to generate
        noise_levels: List of noise levels to apply
    
    Returns:
        Tuple of (bad_trajectories, good_trajectories)
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"Generating {num_optimal_demos} optimal demonstrations...")
    optimal_demos = generate_optimal_demonstrations(
        env, num_demos=num_optimal_demos, max_episode_length=20
    )
    
    bad_trajectories = []
    good_trajectories = []
    
    print("Creating noisy versions...")
    for demo in optimal_demos:
        # Create noisy versions for each noise level
        for i, noise_level in enumerate(noise_levels):
            # Bad trajectory with current noise level
            bad_demo = add_noise_to_trajectory(demo, noise_level)
            bad_trajectories.append(bad_demo)
            
            # Good trajectory with one level less noise (or optimal if at first level)
            if i == 0:
                # For the first noise level, use the optimal demo as the good trajectory
                good_trajectories.append(demo)
            else:
                # Use the previous noise level as the good trajectory
                good_demo = add_noise_to_trajectory(demo, noise_levels[i-1])
                good_trajectories.append(good_demo)
        
        # Add final pair where both good and bad are noise-free (end of improvement chain)
        bad_trajectories.append(demo)  # Noise-free bad trajectory
        good_trajectories.append(demo)  # Noise-free good trajectory (same as bad)
    
    print(f"Generated {len(bad_trajectories)} bad trajectories and {len(good_trajectories)} good trajectories")
    return bad_trajectories, good_trajectories


def train_trajectory_improver(bad_trajectories: List[List[Dict]], 
                             good_trajectories: List[List[Dict]],
                             epochs: int = 100,
                             batch_size: int = 32,
                             learning_rate: float = 1e-3,
                             writer: SummaryWriter = None,
                             max_length: int = 20) -> TrajectoryImproverRNN:
    """
    Train the trajectory improvement RNN.
    
    Args:
        bad_trajectories: List of noisy/bad trajectory demonstrations
        good_trajectories: List of corresponding optimal trajectory demonstrations
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        writer: TensorboardX SummaryWriter for logging
        max_length: Maximum trajectory length for padding
    
    Returns:
        Trained TrajectoryImproverRNN model
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
    
    # Create datasets and dataloaders
    train_dataset = TrajectoryDataset(train_bad, train_good, max_length=max_length)
    val_dataset = TrajectoryDataset(val_bad, val_good, max_length=max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TrajectoryImproverRNN(input_dim=7, hidden_dim=128, num_layers=2, rnn_type='lstm')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training trajectory improver RNN for {epochs} epochs...")
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    print(f"Model architecture: {model.rnn_type.upper()} with {model.num_layers} layers, hidden_dim={model.hidden_dim}")
    
    best_val_loss = float('inf')
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        num_train_batches = 0
        
        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (batch_bad, batch_good) in enumerate(batch_pbar):
            optimizer.zero_grad()
            
            # batch_bad and batch_good are already tensors with shape (batch_size, seq_len, 7)
            predicted_good = model(batch_bad)
            loss = criterion(predicted_good, batch_good)
            
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
            for batch_bad, batch_good in val_dataloader:
                predicted_good = model(batch_bad)
                loss = criterion(predicted_good, batch_good)
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


def improve_trajectory(model: TrajectoryImproverRNN, 
                      trajectory: List[Dict]) -> List[Dict]:
    """
    Improve a single trajectory using the trained RNN model.
    
    Args:
        model: Trained trajectory improvement model
        trajectory: Input trajectory to improve
    
    Returns:
        Improved trajectory
    """
    model.eval()
    improved_trajectory = []
    
    with torch.no_grad():
        # Extract features for entire trajectory at once
        features_list = extract_trajectory_features(trajectory)
        
        # Pad to model's expected length
        max_length = 20  # Should match training max_length
        while len(features_list) < max_length:
            features_list.append(features_list[-1] if features_list else np.zeros(7))
        
        features_list = features_list[:max_length]
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(features_list).unsqueeze(0)  # Shape: (1, seq_len, 7)
        
        # Get model prediction
        predicted_features = model(input_tensor).squeeze(0).numpy()  # Shape: (seq_len, 7)
        
        # Convert back to trajectory format
        for i, step in enumerate(trajectory):
            if i < len(predicted_features):
                improved_step = step.copy()
                improved_step['obs'] = improved_step['obs'].copy()
                improved_step['obs'][:2] = predicted_features[i][:2]  # Update position
                improved_step['action'] = predicted_features[i][4:6]  # Update action
                improved_trajectory.append(improved_step)
            else:
                improved_trajectory.append(step)
    
    return improved_trajectory


def iterated_inference(model: TrajectoryImproverRNN, 
                      trajectory: List[Dict], 
                      num_iterations: int = 5,
                      convergence_threshold: float = 1e-4) -> Tuple[List[Dict], List[List[Dict]]]:
    """
    Apply iterated inference to progressively improve a trajectory.
    
    Args:
        model: Trained trajectory improvement model
        trajectory: Input trajectory to improve
        num_iterations: Maximum number of improvement iterations
        convergence_threshold: Threshold for convergence detection
    
    Returns:
        Tuple of (final improved trajectory, list of intermediate trajectories)
    """
    trajectory_list = [trajectory.copy()]
    current_trajectory = trajectory.copy()
    previous_trajectory = None
    
    print(f"Starting iterated inference with {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        # Improve trajectory
        improved_trajectory = improve_trajectory(model, current_trajectory)
        
        # Check convergence
        if previous_trajectory is not None:
            # Calculate average change in positions
            total_change = 0
            for prev_step, curr_step in zip(previous_trajectory, improved_trajectory):
                pos_change = np.linalg.norm(
                    prev_step['obs'][:2] - curr_step['obs'][:2]
                )
                total_change += pos_change
            
            avg_change = total_change / len(improved_trajectory)
            print(f"Iteration {iteration + 1}: Average position change = {avg_change:.6f}")
            
            if avg_change < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations!")
                break
        
        previous_trajectory = current_trajectory.copy()
        current_trajectory = improved_trajectory
        trajectory_list.append(current_trajectory.copy())
    
    return current_trajectory, trajectory_list


def evaluate_trajectory_quality(trajectory: List[Dict], env: PointMass2DEnv) -> Dict[str, float]:
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
        goal_pos = step['obs'][2:4]  # Goal position is stored in obs[2:4]
        agent_pos = step['obs'][:2]  # Agent position is stored in obs[:2]
        distance = np.linalg.norm(agent_pos - goal_pos)
        reward = -distance  # Reward is negative distance to goal
        total_reward += reward
    
    final_distance = np.linalg.norm(trajectory[-1]['obs'][:2] - trajectory[-1]['obs'][2:4])
    success = final_distance < 0.5  # Assuming success threshold of 0.5
    
    return {
        'total_reward': total_reward,
        'success': success,
        'final_distance': final_distance,
        'num_steps': len(trajectory)
    }


def visualize_trajectory_comparison(env: PointMass2DEnv, 
                                  original_trajectory: List[Dict],
                                  improved_trajectory: List[Dict],
                                  trajectory_list: List[List[Dict]],
                                  save_path: str = None):
    """
    Visualize comparison between original and improved trajectories.
    
    Args:
        env: The pointmass environment
        original_trajectory: Original trajectory
        improved_trajectory: Improved trajectory
        trajectory_list: List of intermediate trajectories
        save_path: Optional path to save visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original trajectory
    ax1.set_title("Original Trajectory")
    ax1.set_xlim(-env.arena_size, env.arena_size)
    ax1.set_ylim(-env.arena_size, env.arena_size)
    ax1.grid(True, alpha=0.3)
    
    # Draw arena boundary
    arena = plt.Rectangle((-env.arena_size, -env.arena_size), 
                         2*env.arena_size, 2*env.arena_size, 
                         fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(arena)
    
    # Plot trajectory
    positions = [step['obs'][:2] for step in original_trajectory]
    positions = np.array(positions)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
    
    # Plot goal
    goal = original_trajectory[0]['obs'][2:4]
    ax1.scatter(goal[0], goal[1], c='orange', s=150, marker='*', label='Goal')
    ax1.legend()
    
    # Plot improved trajectory and intermediates
    ax2.set_title("Improvement Process")
    ax2.set_xlim(-env.arena_size, env.arena_size)
    ax2.set_ylim(-env.arena_size, env.arena_size)
    ax2.grid(True, alpha=0.3)
    
    # Draw arena boundary
    arena = plt.Rectangle((-env.arena_size, -env.arena_size), 
                         2*env.arena_size, 2*env.arena_size, 
                         fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(arena)
    
    # Create color map for trajectory evolution
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_list)))
    
    # Plot intermediate trajectories with color gradient
    for i, (traj, color) in enumerate(zip(trajectory_list, colors)):
        positions = [step['obs'][:2] for step in traj]
        positions = np.array(positions)
        label = f'Iteration {i}' if i == 0 or i == len(trajectory_list)-1 else None
        ax2.plot(positions[:, 0], positions[:, 1], '-', color=color, alpha=0.7, 
                linewidth=1, label=label)
    
    # Plot start and end points
    positions = [step['obs'][:2] for step in improved_trajectory]
    positions = np.array(positions)
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
    
    # Plot goal
    goal = improved_trajectory[0]['obs'][2:4]
    ax2.scatter(goal[0], goal[1], c='orange', s=150, marker='*', label='Goal')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()


def visualize_training_data(env: PointMass2DEnv,
                          bad_trajectories: List[List[Dict]], 
                          good_trajectories: List[List[Dict]], 
                          num_samples: int = 5,
                          save_path: str = None):
    """
    Visualize a sample of training data showing bad and good trajectory pairs.
    
    Args:
        env: The environment
        bad_trajectories: List of noisy/bad trajectories
        good_trajectories: List of corresponding optimal trajectories 
        num_samples: Number of trajectory pairs to visualize
        save_path: Optional path to save the visualization
    """
    # Create figure with subplots for each trajectory pair
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    fig.suptitle("Training Data: Bad vs Good Trajectories", fontsize=16)
    
    # Randomly sample trajectory pairs
    indices = np.random.choice(len(bad_trajectories), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        bad_traj = bad_trajectories[idx]
        good_traj = good_trajectories[idx]
        
        # Plot bad trajectory
        axes[i,0].set_title(f"Bad Trajectory {i+1}")
        axes[i,0].set_xlim(-env.arena_size, env.arena_size)
        axes[i,0].set_ylim(-env.arena_size, env.arena_size)
        axes[i,0].grid(True, alpha=0.3)
        
        # Draw arena boundary
        arena = plt.Rectangle((-env.arena_size, -env.arena_size), 
                            2*env.arena_size, 2*env.arena_size, 
                            fill=False, edgecolor='black', linewidth=2)
        axes[i,0].add_patch(arena)
        
        # Plot trajectory
        positions = [step['obs'][:2] for step in bad_traj]
        positions = np.array(positions)
        axes[i,0].plot(positions[:, 0], positions[:, 1], 'r-', alpha=0.7, linewidth=2)
        axes[i,0].scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
        axes[i,0].scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
        
        # Plot goal
        goal = bad_traj[0]['obs'][2:4]
        axes[i,0].scatter(goal[0], goal[1], c='orange', s=150, marker='*', label='Goal')
        axes[i,0].legend()
        
        # Plot good trajectory
        axes[i,1].set_title(f"Good Trajectory {i+1}")
        axes[i,1].set_xlim(-env.arena_size, env.arena_size)
        axes[i,1].set_ylim(-env.arena_size, env.arena_size)
        axes[i,1].grid(True, alpha=0.3)
        
        # Draw arena boundary
        arena = plt.Rectangle((-env.arena_size, -env.arena_size), 
                            2*env.arena_size, 2*env.arena_size, 
                            fill=False, edgecolor='black', linewidth=2)
        axes[i,1].add_patch(arena)
        
        # Plot trajectory
        positions = [step['obs'][:2] for step in good_traj]
        positions = np.array(positions)
        axes[i,1].plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2)
        axes[i,1].scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
        axes[i,1].scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
        
        # Plot goal
        goal = good_traj[0]['obs'][2:4]
        axes[i,1].scatter(goal[0], goal[1], c='orange', s=150, marker='*', label='Goal')
        axes[i,1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.savefig('trajectory_comparison.png')


def main():
    """Main function to demonstrate trajectory improvement with RNN."""
    
    # Create environment
    print("Creating 2D Point Mass Environment...")
    env = PointMass2DEnv(max_steps=20, dt=0.1, max_delta_pos=0.5)
    
    # Generate training data
    print("\n=== Generating Training Data ===")
    bad_trajectories, good_trajectories = generate_noise_augmented_data(
        env, num_optimal_demos=100, noise_levels=[0.2, 0.4, 0.6, 0.8, 1.0]
    )
    assert len(bad_trajectories) == len(good_trajectories), "Bad and good trajectory lists must have same length"
    for bad_traj, good_traj in zip(bad_trajectories, good_trajectories):
        assert len(bad_traj) == len(good_traj), "Each corresponding bad and good trajectory must have same length"
    
    visualize_training_data(env, bad_trajectories, good_trajectories)

    # Train the trajectory improver
    print("\n=== Training Trajectory Improver RNN ===")
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(f'runs/trajectory_improver_rnn_{datetime.now().strftime("%Y%m%d")}')
    
    # Try to load existing model first
    model_path = 'trajectory_improver_rnn.pt'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = torch.load(model_path)
    else:
        print("Training new model...")
        model = train_trajectory_improver(
            bad_trajectories, good_trajectories, 
            epochs=2000, batch_size=64, learning_rate=1e-3,
            writer=writer, max_length=20
        )
        # Save the trained model
        print(f"Saving model to {model_path}")
        torch.save(model, model_path)
    
    writer.close()

    # Test on a new noisy trajectory
    print("\n=== Testing Trajectory Improvement ===")
    
    # Generate a new optimal demo
    optimal_demo = generate_optimal_demonstrations(env, target=np.array([-3.5, -3.5]), num_demos=1, max_episode_length=20)[0]
    
    # Create a noisy version
    noisy_demo = add_noise_to_trajectory(optimal_demo, noise_level=1.5)
    
    # Evaluate original noisy trajectory
    original_quality = evaluate_trajectory_quality(noisy_demo, env)
    print(f"Original trajectory quality: {original_quality}")
    
    # Apply iterated inference
    improved_demo, trajectory_list = iterated_inference(model, noisy_demo, num_iterations=5)
    
    # Evaluate improved trajectory
    improved_quality = evaluate_trajectory_quality(improved_demo, env)
    print(f"Improved trajectory quality: {improved_quality}")
    
    # Visualize comparison
    print("\n=== Visualizing Results ===")
    visualize_trajectory_comparison(env, noisy_demo, improved_demo, trajectory_list, save_path='plots/trajectory_comparison_improvement_rnn.png')
    
    # Test with a completely random trajectory
    print("\n=== Testing with Random Trajectory ===")
    random_trajectory = []
    obs, info = env.reset()
    
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        random_trajectory.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward,
            'done': done,
            'info': info.copy()
        })
        
        if done:
            break
    
    random_quality = evaluate_trajectory_quality(random_trajectory, env)
    print(f"Random trajectory quality: {random_quality}")
    
    improved_random, trajectory_list = iterated_inference(model, random_trajectory, num_iterations=5)
    improved_random_quality = evaluate_trajectory_quality(improved_random, env)
    print(f"Improved random trajectory quality: {improved_random_quality}")
    
    # Visualize random trajectory improvement
    visualize_trajectory_comparison(env, random_trajectory, improved_random, trajectory_list, save_path='plots/random_trajectory_improvement_rnn.png')
    
    env.close()
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main() 