import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os


class PointMassWithWallsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 60}

    def __init__(self, size=500, render_mode=None, max_steps=500):
        super().__init__()
        self.size = size
        self.render_mode = render_mode
        self.max_steps = max_steps  # ✅ Set maximum episode length
        self.step_count = 0  # ✅ Track steps

        self.screen = None
        self.clock = None

        self.walls = [
            # (self.size // 3, 100, self.size // 3, 400),  # ✅ Vertical wall at 1/3
            # (2 * self.size // 3, 150, 2 * self.size // 3, 350),  # ✅ Vertical wall at 2/3
            # (self.size // 2, 50, self.size // 2, 450),  # ✅ Vertical wall in center
            (150, 200, 500,200),  # ✅ Horizontal wall in middle
        ]

        if self.render_mode in ["human", "rgb_array"]:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.size, self.size))  # Window mode
            self.clock = pygame.time.Clock()

        # ✅ Keep original action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ✅ Observation space is now normalized between -1 and 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ✅ Original state (not normalized)
        #self.initial_state = np.array([250.0, 20.0], dtype=np.float32)
        self.initial_state = np.array([50.0, 180.0], dtype=np.float32)

        self.state = np.array(self.initial_state, dtype=np.float32)

        self.goal_state = np.array([self.size - 100, self.size - 50], dtype=np.float32)
        self.goal_tol = 15

    def normalize_obs(self, obs):
        """ ✅ Normalize observations from [0, size] → [-1, 1] """
        return (obs - (self.size / 2)) / (self.size / 2)  # Center around 0

    def denormalize_action(self, action):
        """ ✅ Convert actions from [-1, 1] → environment scale """
        return action * 5 # Scale actions to make them effective

    def step(self, action):
        """
        Applies an action, updates the state, and returns a normalized observation.
        """
        distance = np.linalg.norm(self.goal_state - self.state)

        # ✅ Scale action to real-world movement
        action = self.denormalize_action(action)

        next_state = self.state + action  # ✅ Apply movement
        next_x, next_y = next_state

        blocked = False  # ✅ Track if movement is blocked

        # ✅ Check all walls BEFORE updating state
        for x_start, y_start, x_end, y_end in self.walls:
            if x_start == x_end:  # ✅ Vertical wall
                if (x_start - 5 <= next_x <= x_start + 5) and (y_start <= next_y <= y_end):
                    print(f"🚧 Agent hit vertical wall at x={x_start}! Cancelling movement.")
                    blocked = True

            if y_start == y_end:  # ✅ Horizontal wall
                if (y_start - 5 <= next_y <= y_start + 5) and (x_start <= next_x <= x_end):
                    #print(f"🚧 Agent hit horizontal wall at y={y_start}! Cancelling movement.")
                    blocked = True

        # ✅ Only update state if movement is not blocked
        if not blocked:
            self.state = np.clip(next_state, 10.0, self.size - 10.0)  # ✅ Keep inside boundaries

        self.step_count += 1  # ✅ Increment step counter

        new_distance = np.linalg.norm(self.goal_state - self.state)

        # ✅ Give a reward for moving closer to the goal
        reward = (distance - new_distance) * 5  # ✅ Encourage movement
        reward -= 0.1  # Small step penalty
        if new_distance <= self.goal_tol:
            print('Goal Reached!')
            reward += 100
        corner_threshold = 15
        corners = [
            np.array([10, 10]),  # Bottom-left
            np.array([10, self.size - 10]),  # Top-left
            np.array([self.size - 10, 10]),  # Bottom-right
            np.array([self.size - 10, self.size - 10])  # ✅ Top-right (fixes missing corner)
        ]

        for corner in corners:
            if np.linalg.norm(self.state - corner) < corner_threshold:
                #print(f"🚨 Agent Stuck in Corner {corner}! Applying Penalty")
                reward -= 300  # ✅ Apply penalty

        # ✅ Episode ends if goal is reached OR max steps reached
        done = new_distance <= self.goal_tol or self.step_count >= self.max_steps

        # ✅ Normalize observation before returning
        normalized_obs = self.normalize_obs(self.state)

        if self.render_mode == "human":
            self.render()

        return normalized_obs, reward, done, False, {"distance": distance, "steps": self.step_count}

    def reset(self, seed=None, options=None):
        """
        Resets the environment, including the step counter.
        """
        #print(f"🚨 Reset: Last state was {self.state}")
        super().reset(seed=seed)
        self.state = np.array(self.initial_state, dtype=np.float32)  # ✅ Reset to initial state
        self.step_count = 0  # ✅ Reset step counter

        # ✅ Return normalized observation
        normalized_obs = self.normalize_obs(self.state)
        return normalized_obs, {}

    def render(self):
        """
        Handles Pygame rendering, ensuring the window updates properly.
        Works for both standalone testing and ClearRL DDPG integration.
        """
        if self.render_mode is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if self.screen is None and self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.size, self.size))

        if self.render_mode == "rgb_array":
            surface = pygame.Surface((self.size, self.size))  # ✅ Render to an off-screen surface
        else:
            surface = self.screen  # ✅ Use actual window

        surface.fill((255, 255, 255))  # White background

        pygame.draw.rect(
            surface,
            (0, 200, 0),
            pygame.Rect(self.goal_state[0] - self.goal_tol, self.goal_state[1] - self.goal_tol, 20, 20),
        )

        pygame.draw.circle(
            surface,
            (0, 0, 255),
            self.state.astype(int),
            10
        )

        for x_start, y_start, x_end, y_end in self.walls:
            pygame.draw.line(
                surface,
                (0, 0, 0),  # ✅ Black wall
                (x_start, y_start),
                (x_end, y_end),
                5  # ✅ Wall thickness
            )


        if self.render_mode == "human":
            pygame.display.flip()  # ✅ Update the window
            self.clock.tick(self.metadata["render_fps"])  # ✅ Maintain FPS
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(surface).swapaxes(0, 1)  # ✅ Returns frame for RL algorithms

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
