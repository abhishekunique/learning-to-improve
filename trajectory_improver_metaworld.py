#!/usr/bin/env python3
"""
Trajectory Improvement using MLP with Noise-Augmented Demonstrations

This module implements:
1. Generate optimal demonstrations from 2D pointmass environment
2. Add increasing noise to create progressively worse demonstrations
3. Train an MLP to improve trajectories
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
from expert_data_gen import get_expert_trajectory, execute_policy
import metaworld
import gymnasium as gym

def extract_step_features(step: Dict[str, Any]) -> np.ndarray:
    """
    Extract features from a single trajectory step.
    
    Args:
        step: Dictionary containing trajectory step data with 'obs' and 'action' keys
    
    Returns:
        Feature array: [pos_x, pos_y, goal_x, goal_y, action_x, action_y, l2_distance_to_goal]
    """
    pos = step['obs'][:3]
    goal = step['obs'][3:6]
    l2_distance = np.linalg.norm(pos - goal)
    
    features = np.concatenate([
        step['obs'][:6],  # pos and goal
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
    """Dataset for trajectory improvement training."""
    
    def __init__(self, bad_trajectories: List[List[Dict]], good_trajectories: List[List[Dict]]):
        """
        Args:
            bad_trajectories: List of noisy/bad trajectory demonstrations
            good_trajectories: List of corresponding optimal trajectory demonstrations
        """
        self.bad_trajectories = []
        self.good_trajectories = []
        
        # Process each trajectory pair
        for bad_traj, good_traj in zip(bad_trajectories, good_trajectories):
            # Pad shorter trajectory with last state
            max_len = max(len(bad_traj), len(good_traj))
            
            # Extract features for entire trajectory
            bad_features = []
            good_features = []
            
            for i in range(max_len):
                bad_step = bad_traj[min(i, len(bad_traj) - 1)]
                good_step = good_traj[min(i, len(good_traj) - 1)]
                
                # Extract features using the utility function
                bad_step_features = extract_step_features(bad_step)
                good_step_features = extract_step_features(good_step)
                
                bad_features.append(bad_step_features)
                good_features.append(good_step_features)
            
            # Store flattened trajectory features
            # Flatten features into single vector for each trajectory
            flattened_bad = np.array(bad_features).flatten()
            flattened_good = np.array(good_features).flatten()
            self.bad_trajectories.append(flattened_bad)
            self.good_trajectories.append(flattened_good)
    
    def __len__(self):
        return len(self.bad_trajectories)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.bad_trajectories[idx]),
            torch.FloatTensor(self.good_trajectories[idx])
        )


class TrajectoryImproverMLP(nn.Module):
    """MLP model for improving trajectories."""
    
    def __init__(self, input_dim: int = 550, hidden_dim: int = 128, output_dim: int = 550, num_layers: int = 4):
        super().__init__()
        
        # Build network layers dynamically
        layers = []
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:  # Don't add activation after last layer
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


def add_noise_to_trajectory(env, env_name, trajectory: List[Dict], noise_level: float, max_episode_length: int = 50) -> List[Dict]:
    """
    Add noise to a trajectory to create a worse version.
    
    Args:
        trajectory: List of trajectory steps
        noise_level: Standard deviation of noise to add
    
    Returns:
        Noisy trajectory
    """
    # Extract goal from first step of trajectory
    original_goal = trajectory[0]['obs'][3:6]
    
    # Add noise to goal
    goal_noise = np.random.uniform(-noise_level, noise_level, 3)
    noisy_goal = original_goal + goal_noise

    noisy_trajectory = get_expert_trajectory(env, env_name, goal_pos=noisy_goal, max_traj_len=max_episode_length, overwrite_goal=original_goal)

    return noisy_trajectory


def generate_noise_augmented_data(env: PointMass2DEnv, env_name: str,
                                 num_optimal_demos: int = 50,
                                 noise_levels: List[float] = None,
                                 max_episode_length: int = 50) -> Tuple[List, List]:
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
    optimal_demos = [get_expert_trajectory(env, env_name, max_traj_len=max_episode_length) for _ in range(num_optimal_demos)]
    
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
    return bad_trajectories, good_trajectories


def train_trajectory_improver(bad_trajectories: List[List[Dict]], 
                             good_trajectories: List[List[Dict]],
                             epochs: int = 100,
                             batch_size: int = 32,
                             learning_rate: float = 1e-3,
                             writer: SummaryWriter = None) -> TrajectoryImproverMLP:
    """
    Train the trajectory improvement MLP.
    
    Args:
        bad_trajectories: List of noisy/bad trajectory demonstrations
        good_trajectories: List of corresponding optimal trajectory demonstrations
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        writer: TensorboardX SummaryWriter for logging
    
    Returns:
        Trained TrajectoryImproverMLP model
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
    train_dataset = TrajectoryDataset(train_bad, train_good)
    val_dataset = TrajectoryDataset(val_bad, val_good)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TrajectoryImproverMLP(input_dim=550, hidden_dim=1024, output_dim=550, num_layers=6)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training trajectory improver for {epochs} epochs...")
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
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


def improve_trajectory(model: TrajectoryImproverMLP, 
                      trajectory: List[Dict]) -> List[Dict]:
    """
    Improve a single trajectory using the trained model.
    
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
            
        # Stack features into batch tensor
        input_tensor = torch.FloatTensor(np.stack(features_list)).view(1, -1) # Add batch dimension of size 1
        
        # Get model predictions for entire trajectory
        predicted_features = model(input_tensor).detach().numpy().reshape(-1, 11)
        
        # Create improved trajectory
        for i, step in enumerate(trajectory):
            improved_step = step.copy()
            improved_step['obs'] = improved_step['obs'].copy()
            improved_step['obs'][:3] = predicted_features[i,:3]  # Update position
            improved_step['action'] = predicted_features[i,6:10]  # Update action
            improved_trajectory.append(improved_step)
    
    return improved_trajectory


def iterated_inference(model: TrajectoryImproverMLP, 
                      trajectory: List[Dict], 
                      num_iterations: int = 5,
                      convergence_threshold: float = 1e-4) -> List[Dict]:
    """
    Apply iterated inference to progressively improve a trajectory.
    
    Args:
        model: Trained trajectory improvement model
        trajectory: Input trajectory to improve
        num_iterations: Maximum number of improvement iterations
        convergence_threshold: Threshold for convergence detection
    
    Returns:
        Progressively improved trajectory
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
                    prev_step['obs'][:3] - curr_step['obs'][:3]
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
        goal_pos = step['obs'][3:6]  # Goal position is stored in obs[2:4]
        agent_pos = step['obs'][:3]  # Agent position is stored in obs[:2]
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


def visualize_trajectory_comparison(env: PointMass2DEnv, 
                                  original_trajectory: List[Dict],
                                  improved_trajectory: List[Dict],
                                  trajectory_list: List[List[Dict]],
                                  overall_mins: np.ndarray,
                                  overall_maxs: np.ndarray,
                                  save_path: str = None):
    """
    Visualize comparison between original and improved trajectories in 3D.
    
    Args:
        env: The pointmass environment
        original_trajectory: Original trajectory
        improved_trajectory: Improved trajectory
        trajectory_list: List of intermediate trajectories
        save_path: Optional path to save visualization
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(20, 8))
    
    # Create 3D subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot original trajectory in 3D
    ax1.set_title("Original Trajectory (3D)", fontsize=14)
    ax1.set_xlim(overall_mins[0], overall_maxs[0])
    ax1.set_ylim(overall_mins[1], overall_maxs[1])
    ax1.set_zlim(overall_mins[2], overall_maxs[2])
    ax1.grid(True, alpha=0.3)
    
    # Extract 3D positions (x, y, time_step)
    positions_3d = []
    for i, step in enumerate(original_trajectory):
        pos_2d = step['obs'][:3]  # Get x, y, z position
        positions_3d.append([pos_2d[0], pos_2d[1], pos_2d[2]])  # Add time step as z-coordinate
    
    positions_3d = np.array(positions_3d)
    
    # Plot 3D trajectory
    ax1.plot(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2], 
             'b-', alpha=0.8, linewidth=3, label='Trajectory')
    
    # Plot start and end points
    ax1.scatter(positions_3d[0, 0], positions_3d[0, 1], positions_3d[0, 2], 
                c='green', s=200, label='Start', marker='o')
    ax1.scatter(positions_3d[-1, 0], positions_3d[-1, 1], positions_3d[-1, 2], 
                c='red', s=200, label='End', marker='o')
    
    # Plot goal in 3D (projected to all time steps)
    goal = original_trajectory[0]['obs'][3:6]
    ax1.scatter(goal[0], goal[1], goal[2], c='orange', s=200, label='Goal', marker='*')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Time Step')
    ax1.legend()
    
    # Plot improved trajectory and intermediates in 3D
    ax2.set_title("Improvement Process (3D)", fontsize=14)
    ax2.set_xlim(overall_mins[0], overall_maxs[0])
    ax2.set_ylim(overall_mins[1], overall_maxs[1])
    ax2.set_zlim(overall_mins[2], overall_maxs[2])
    ax2.grid(True, alpha=0.3)
    
    # Create color map for trajectory evolution
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_list)))
    
    # Plot intermediate trajectories with color gradient
    for i, (traj, color) in enumerate(zip(trajectory_list, colors)):
        positions_3d = []
        for j, step in enumerate(traj):
            pos_2d = step['obs'][:3]
            positions_3d.append([pos_2d[0], pos_2d[1], pos_2d[2]])
        
        positions_3d = np.array(positions_3d)
        label = f'Iteration {i}'
        ax2.plot(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2], 
                '-', color=color, alpha=0.6, linewidth=2, label=label)
    
    # Plot start and end points of final improved trajectory
    final_positions_3d = []
    for i, step in enumerate(improved_trajectory):
        pos_2d = step['obs'][:3]
        final_positions_3d.append([pos_2d[0], pos_2d[1], pos_2d[2]])
    
    final_positions_3d = np.array(final_positions_3d)
    ax2.scatter(final_positions_3d[0, 0], final_positions_3d[0, 1], final_positions_3d[0, 2], 
                c='green', s=200, label='Start', marker='o')
    ax2.scatter(final_positions_3d[-1, 0], final_positions_3d[-1, 1], final_positions_3d[-1, 2], 
                c='red', s=200, label='End', marker='o')
    
    # Plot goal in 3D
    goal = improved_trajectory[0]['obs'][3:6]
    ax2.scatter(goal[0], goal[1], goal[2], c='orange', s=200, label='Goal', marker='*')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Time Step')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_training_data(env: PointMass2DEnv,
                          bad_trajectories: List[List[Dict]], 
                          good_trajectories: List[List[Dict]], 
                          overall_mins: np.ndarray,
                          overall_maxs: np.ndarray,
                          num_samples: int = 5,
                          save_path: str = None):
    """
    Visualize a sample of training data showing bad and good trajectory pairs in 3D.
    
    Args:
        env: The environment
        bad_trajectories: List of noisy/bad trajectories
        good_trajectories: List of corresponding optimal trajectories 
        num_samples: Number of trajectory pairs to visualize
        save_path: Optional path to save the visualization
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with 3D subplots
    fig = plt.figure(figsize=(20, 4*num_samples))
    fig.suptitle("Training Data: Bad vs Good Trajectories (3D)", fontsize=16)
    
    # Randomly sample trajectory pairs
    indices = np.random.choice(len(bad_trajectories), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        bad_traj = bad_trajectories[idx]
        good_traj = good_trajectories[idx]
        
        # Create 3D subplots for this pair
        ax1 = fig.add_subplot(num_samples, 2, 2*i + 1, projection='3d')
        ax2 = fig.add_subplot(num_samples, 2, 2*i + 2, projection='3d')
        
        # Plot bad trajectory in 3D
        ax1.set_title(f"Bad Trajectory {i+1} (3D)", fontsize=12)
        ax1.set_xlim(overall_mins[0], overall_maxs[0])
        ax1.set_ylim(overall_mins[1], overall_maxs[1])
        ax1.set_zlim(overall_mins[2], overall_maxs[2])
        ax1.grid(True, alpha=0.3)
        
        # Extract 3D positions for bad trajectory
        bad_positions_3d = []
        for step in bad_traj:
            pos_3d = step['obs'][:3]  # Already 3D coordinates
            bad_positions_3d.append(pos_3d)
        
        bad_positions_3d = np.array(bad_positions_3d)
        
        # Plot 3D trajectory
        ax1.plot(bad_positions_3d[:, 0], bad_positions_3d[:, 1], bad_positions_3d[:, 2], 
                 'r-', alpha=0.8, linewidth=3, label='Bad Trajectory')
        
        # Plot start and end points
        ax1.scatter(bad_positions_3d[0, 0], bad_positions_3d[0, 1], bad_positions_3d[0, 2], 
                    c='green', s=150, label='Start', marker='o')
        ax1.scatter(bad_positions_3d[-1, 0], bad_positions_3d[-1, 1], bad_positions_3d[-1, 2], 
                    c='red', s=150, label='End', marker='o')
        
        # Plot goal in 3D
        goal = bad_traj[0]['obs'][3:6]  # 3D goal
        ax1.scatter(goal[0], goal[1], goal[2], 
                   c='orange', s=150, label='Goal', marker='*')
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')
        ax1.legend()
        
        # Plot good trajectory in 3D
        ax2.set_title(f"Good Trajectory {i+1} (3D)", fontsize=12)
        ax2.set_xlim(overall_mins[0], overall_maxs[0])
        ax2.set_ylim(overall_mins[1], overall_maxs[1])
        ax2.set_zlim(overall_mins[2], overall_maxs[2])
        ax2.grid(True, alpha=0.3)
        
        # Extract 3D positions for good trajectory
        good_positions_3d = []
        for step in good_traj:
            pos_3d = step['obs'][:3]  # Already 3D coordinates
            good_positions_3d.append(pos_3d)
        
        good_positions_3d = np.array(good_positions_3d)
        
        # Plot 3D trajectory
        ax2.plot(good_positions_3d[:, 0], good_positions_3d[:, 1], good_positions_3d[:, 2], 
                 'b-', alpha=0.8, linewidth=3, label='Good Trajectory')
        
        # Plot start and end points
        ax2.scatter(good_positions_3d[0, 0], good_positions_3d[0, 1], good_positions_3d[0, 2], 
                    c='green', s=150, label='Start', marker='o')
        ax2.scatter(good_positions_3d[-1, 0], good_positions_3d[-1, 1], good_positions_3d[-1, 2], 
                    c='red', s=150, label='End', marker='o')
        
        # Plot goal in 3D
        goal = good_traj[0]['obs'][3:6]  # 3D goal
        ax2.scatter(goal[0], goal[1], goal[2],
                   c='orange', s=150, label='Goal', marker='*')
        
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_zlabel('Z Position')
        ax2.legend()
    
    plt.tight_layout()

    plt.savefig('trajectory_comparison.png')

def main():
    """Main function to demonstrate trajectory improvement."""
    
    # Create environment
    print("Creating Reaching Environment...")
    env_name = "reach-v3"
    max_traj_len = 50
    env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array")
    
    # Generate training data
    print("\n=== Generating Training Data ===")
    bad_trajectories, good_trajectories = generate_noise_augmented_data(
        env, env_name, num_optimal_demos=5, noise_levels=[0.2, 0.4, 0.6, 0.8, 1.0], max_episode_length=max_traj_len
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
    print("\n=== Training Trajectory Improver ===")
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(f'runs/trajectory_improver_{datetime.now().strftime("%Y%m%d")}')
    # Try to load existing model first
    model_path = 'models/trajectory_improver_largerrange.pt'
    load_model = True
    if load_model and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = torch.load(model_path)
    else:
        print("Training new model...")
        model = train_trajectory_improver(
            bad_trajectories, good_trajectories, 
            epochs=60, batch_size=128, learning_rate=1e-3,
            writer=writer
        )
        # Save the trained model
        print(f"Saving model to {model_path}")
        torch.save(model, model_path)
    
    writer.close()

    # Test on a new noisy trajectory
    print("\n=== Testing Trajectory Improvement ===")
    num_eval_trajectories = 20
    improvements_noisy = []
    improvements_random = []
    for i in range(num_eval_trajectories):
        # Evaluate on a new noisy trajectory
        optimal_demo = get_expert_trajectory(env, env_name, max_traj_len=max_traj_len)
        noisy_demo = add_noise_to_trajectory(env, env_name, optimal_demo, noise_level=1.0)
        original_quality = evaluate_trajectory_quality(noisy_demo, env)
        print(f"Original trajectory quality: {original_quality}")
        improved_demo, trajectory_list = iterated_inference(model, noisy_demo, num_iterations=10)
        improved_quality = evaluate_trajectory_quality(improved_demo, env)
        print(f"Improved trajectory quality: {improved_quality}")
        improvements_noisy.append(improved_quality['final_distance'] - original_quality['final_distance'])
        visualize_trajectory_comparison(env, noisy_demo, improved_demo, trajectory_list, save_path=f'plots/mw_trajectory_comparison_improvement_{i}.png', overall_mins=overall_mins, overall_maxs=overall_maxs)

        # Evaluate on a new random trajectory
        random_demo = get_expert_trajectory(env, env_name, max_traj_len=max_traj_len, random_actions=True)
        random_quality = evaluate_trajectory_quality(random_demo, env)
        print(f"Random trajectory quality: {random_quality}")
        improved_random, trajectory_list = iterated_inference(model, random_demo, num_iterations=10)
        improved_random_quality = evaluate_trajectory_quality(improved_random, env)
        print(f"Improved random trajectory quality: {improved_random_quality}")
        improvements_random.append(improved_random_quality['final_distance'] - random_quality['final_distance'])
        visualize_trajectory_comparison(env, random_demo, improved_random, trajectory_list, save_path=f'plots/mw_trajectory_comparison_improvement_random_{i}.png', overall_mins=overall_mins, overall_maxs=overall_maxs)

    print(f"Average improvement for noisy trajectories: {np.mean(improvements_noisy)}")
    print(f"Average improvement for random trajectories: {np.mean(improvements_random)}")

if __name__ == "__main__":
    main() 