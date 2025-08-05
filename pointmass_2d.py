import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random


class PointMass2DEnv(gym.Env):
    """
    2D Point Mass Environment with goal conditioning.
    The agent controls a point mass in 2D space to reach a goal position.
    """
    
    def __init__(self, 
                 max_steps: int = 100,
                 dt: float = 0.1,
                 max_delta_pos: float = 0.5,
                 goal_threshold: float = 0.1,
                 arena_size: float = 5.0,
                 goal_range: float = 3.0):
        super().__init__()
        
        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt
        self.max_delta_pos = max_delta_pos
        self.goal_threshold = goal_threshold
        self.arena_size = arena_size
        self.goal_range = goal_range
        
        # State variables
        self.pos = np.zeros(2)  # position (x, y)
        self.goal = np.zeros(2)  # goal position
        self.step_count = 0
        
        # Action space: continuous delta position in x and y directions
        self.action_space = spaces.Box(
            low=-max_delta_pos, high=max_delta_pos, shape=(2,), dtype=np.float32
        )
        
        # Observation space: position and goal
        # [pos_x, pos_y, goal_x, goal_y]
        obs_low = np.array([-arena_size, -arena_size, -goal_range, -goal_range])
        obs_high = np.array([arena_size, arena_size, goal_range, goal_range])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        
        # Rendering
        self.fig = None
        self.ax = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset position to (0, 0)
        self.pos = np.zeros(2)
        
        # Sample a random goal within the goal range
        self.goal = self.np_random.uniform(
            low=-self.goal_range, high=self.goal_range, size=(2,)
        )
        
        # Ensure goal is within arena bounds
        self.goal = np.clip(self.goal, -self.arena_size + 0.5, self.arena_size - 0.5)
        
        self.step_count = 0
        
        obs = self._get_obs()
        info = {
            'goal': self.goal.copy(),
            'pos': self.pos.copy()
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Clip action to valid range
        action = np.clip(action, -self.max_delta_pos, self.max_delta_pos)
        
        # Update position directly (delta position control)
        self.pos += action
        
        # Clip position to arena bounds
        self.pos = np.clip(self.pos, -self.arena_size, self.arena_size)
        
        self.step_count += 1
        
        # Calculate reward
        distance_to_goal = np.linalg.norm(self.pos - self.goal)
        reward = self._compute_reward(distance_to_goal, action)
        
        # Check if episode is done
        done = (distance_to_goal < self.goal_threshold) or (self.step_count >= self.max_steps)
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_obs()
        info = {
            'goal': self.goal.copy(),
            'pos': self.pos.copy(),
            'distance_to_goal': distance_to_goal,
            'success': distance_to_goal < self.goal_threshold
        }
        
        return obs, reward, done, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([self.pos, self.goal]).astype(np.float32)
    
    def _compute_reward(self, distance_to_goal: float, action: np.ndarray) -> float:
        """Compute reward based on distance to goal and action cost."""
        # Reward for being close to goal
        goal_reward = -distance_to_goal
        
        # Penalty for using too much movement (action cost)
        action_cost = 0.0
        
        # Bonus for reaching goal
        if distance_to_goal < self.goal_threshold:
            goal_reward += 10.0
        
        return goal_reward + action_cost
    
    def render(self, mode='human'):
        """Render the environment."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.ax.clear()
        
        # Set up the plot
        self.ax.set_xlim(-self.arena_size, self.arena_size)
        self.ax.set_ylim(-self.arena_size, self.arena_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw arena boundary
        arena = plt.Rectangle((-self.arena_size, -self.arena_size), 
                            2*self.arena_size, 2*self.arena_size, 
                            fill=False, edgecolor='black', linewidth=2)
        self.ax.add_patch(arena)
        
        # Draw agent (blue circle)
        agent = Circle(self.pos, 0.1, color='blue', alpha=0.7)
        self.ax.add_patch(agent)
        
        # Draw goal (red circle)
        goal = Circle(self.goal, 0.15, color='red', alpha=0.7)
        self.ax.add_patch(goal)
        
        # Draw action vector (if available)
        # Note: In delta position control, we don't have velocity to display
        
        self.ax.set_title(f'Step: {self.step_count}, Distance: {np.linalg.norm(self.pos - self.goal):.3f}')
        
        plt.pause(0.01)
    
    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def generate_optimal_demonstrations(env: PointMass2DEnv,
                                  target: Optional[np.ndarray] = None,
                                  num_demos: int = 10, 
                                  max_episode_length: int = 100) -> list:
    """
    Generate optimal demonstrations for the 2D pointmass environment.
    
    Args:
        env: The PointMass2DEnv environment
        num_demos: Number of demonstrations to generate
        max_episode_length: Maximum length of each demonstration
    
    Returns:
        List of demonstrations, each containing (obs, action, reward, done, info)
    """
    demonstrations = []
    
    for demo_idx in range(num_demos):
        print(f"Generating demonstration {demo_idx + 1}/{num_demos}")
        
        obs, info = env.reset()
        if target is not None:
            env.goal = target
        obs = env._get_obs()
        demo = []
        
        for step in range(max_episode_length):
            # Compute optimal action using simple proportional control
            pos = obs[:2]
            goal = target if target is not None else obs[2:4]
            
            # Distance to goal
            distance_to_goal = np.linalg.norm(pos - goal)
            
            if distance_to_goal < env.goal_threshold:
                # Goal reached, stop
                action = np.zeros(2)
            else:
                # Direct proportional control towards goal
                # Compute desired delta position
                desired_delta = (goal - pos) * 0.5  # Proportional gain
                
                # Clip to maximum delta position
                action = np.clip(desired_delta, -env.max_delta_pos, env.max_delta_pos)
            
            # Take step
            next_obs, reward, done, truncated, step_info = env.step(action)
            
            # Store transition
            demo.append({
                'obs': obs.copy(),
                'action': action.copy(),
                'reward': reward,
                'done': done,
                'info': step_info.copy()
            })
            
            obs = next_obs
            
            # if done:
            #     break
        
        demonstrations.append(demo)
        print(f"  Demo {demo_idx + 1} completed in {len(demo)} steps")
    
    return demonstrations


def visualize_demonstration(env: PointMass2DEnv, demo: list, save_path: Optional[str] = None):
    """
    Visualize a demonstration by replaying it.
    
    Args:
        env: The PointMass2DEnv environment
        demo: List of demonstration steps
        save_path: Optional path to save the visualization
    """
    env.reset()
    
    # Set the goal to match the demonstration
    if demo:
        goal = demo[0]['info']['goal']
        env.goal = goal
    
    for step_data in demo:
        obs = step_data['obs']
        action = step_data['action']
        
        # Update environment state to match demonstration
        env.pos = obs[:2]
        env.goal = obs[2:4]
        
        # Render
        env.render()
        
        # Small delay to see the movement
        plt.pause(0.1)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def test_environment():
    """Test the environment and demonstration generation."""
    # Create environment
    env = PointMass2DEnv(max_steps=100, dt=0.1, max_delta_pos=0.5)
    
    print("Testing environment...")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Goal: {info['goal']}")
    
    # Test random actions
    print("\nTesting random actions...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: pos={env.pos}, reward={reward:.3f}")
        
        if done:
            break
    
    # Generate demonstrations
    print("\nGenerating optimal demonstrations...")
    demos = generate_optimal_demonstrations(env, num_demos=3, max_episode_length=50)
    
    # Visualize first demonstration
    print("\nVisualizing first demonstration...")
    visualize_demonstration(env, demos[0])
    
    env.close()


if __name__ == "__main__":
    test_environment() 