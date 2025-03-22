import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os


class CustomPointMassEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 60}

    def __init__(self, size=500, render_mode=None, max_steps=500, fixed_start=False):
        super().__init__()
        self.size = size
        self.render_mode = render_mode
        self.max_steps = max_steps  # ✅ Set maximum episode length
        self.step_count = 0  # ✅ Track steps
        self.fixed_start = fixed_start
        self.screen = None
        self.clock = None
        self.fixed_state = np.array([250.0, 20.0], dtype=np.float32)

        self.walls = [
            # (self.size // 3, 100, self.size // 3, 400),  # ✅ Vertical wall at 1/3
            # (2 * self.size // 3, 150, 2 * self.size // 3, 350),  # ✅ Vertical wall at 2/3
            # (self.size // 2, 50, self.size // 2, 450),  # ✅ Vertical wall in center
            #(150, 200, 500,200),  # ✅ Horizontal wall in middle
        ]

        if self.render_mode in ["human", "rgb_array"]:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.size, self.size))  # Window mode
            self.clock = pygame.time.Clock()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, 0.0, -np.inf], dtype=np.float32),
            high=np.array([self.size, np.inf, self.size, np.inf], dtype=np.float32),
        )

        self.initial_state = np.array(self.fixed_state, dtype=np.float32)
        self.state = np.array([self.initial_state[0], 0.0, self.initial_state[1], 0.0], dtype=np.float32)

        self.goal_state = np.array([self.size - 100, self.size - 50], dtype=np.float32)
        self.goal_tol = 15

        self._dt = 0.1
        self._friction = 0.2
        self._prot_mass = 1.0

    def step(self, action):
        """
        Applies an action, updates the state, and returns a normalized observation.
        """
        distance = np.linalg.norm(self.goal_state - self.state[[0, 2]])

        action = self.denormalize_action(action)
        next_state = self._step_internal(self.state, action)  # ✅ Apply movement
        next_x, next_y = next_state[0], next_state[2]

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

        if not blocked:
            self.state = next_state

        self.step_count += 1  # ✅ Increment step counter

        new_distance = np.linalg.norm(self.goal_state - self.state[[0, 2]])

        reward = (distance - new_distance) * 5  # ✅ Encourage movement
        if distance - new_distance == 0.0:
            reward -= 100

        if new_distance <= self.goal_tol:
            print('Goal Reached!')
            reward += 100

        corner_threshold = 15
        corners = [
            np.array([10, 10]),
            np.array([10, self.size - 10]),
            np.array([self.size - 10, 10]),
            np.array([self.size - 10, self.size - 10])
        ]

        for corner in corners:
            if np.linalg.norm(self.state[[0, 2]] - corner) < corner_threshold:
                #print(f"🚨 Agent Stuck in Corner {corner}! Applying Penalty")
                reward -= 300  # ✅ Apply penalty

        done = new_distance <= self.goal_tol or self.step_count >= self.max_steps


        if self.render_mode == "human":
            self.render()

        normalized_obs = self.normalize_obs(self.state)
        return normalized_obs, reward, done, False, {"distance": distance, "steps": self.step_count}

    def reset(self, seed=None, options=None):
        """
        Resets the environment, including the step counter.
        """
        super().reset(seed=seed)
        if self.fixed_start:
            self.initial_state = np.array(self.fixed_state, dtype=np.float32)
        else:
            if np.random.rand() < 0.5:
                self.initial_state = np.array(self.fixed_state, dtype=np.float32)
            else:
                self.initial_state = np.array(self.get_random_start_position(), dtype=np.float32)

        self.state = np.array([self.initial_state[0], 0.0, self.initial_state[1], 0.0], dtype=np.float32)
        self.step_count = 0  # ✅ Reset step counter

        normalized_obs = self.normalize_obs(self.state)
        return normalized_obs, {}

    def _step_internal(self, state, action):
        state_der = np.zeros(4)
        state_der[0] = state[1]
        state_der[1] = (action[0] - self._friction * state[1]) / self._prot_mass
        state_der[2] = state[3]
        state_der[3] = (action[1] - self._friction * state[3]) / self._prot_mass

        new_state = state + self._dt * state_der

        if new_state[0] < 10 or new_state[0] > self.size - 10:
            new_state[1] = 0
        if new_state[2] < 10 or new_state[2] > self.size - 10:
            new_state[3] = 0

        new_state[0] = np.clip(new_state[0], 10, self.size - 10)
        new_state[2] = np.clip(new_state[2], 10, self.size - 10)

        return new_state

    def render(self):
        if self.render_mode is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if self.screen is None and self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.size, self.size))

        if self.render_mode == "rgb_array":
            surface = pygame.Surface((self.size, self.size))
        else:
            surface = self.screen

        surface.fill((255, 255, 255))

        pygame.draw.rect(
            surface,
            (0, 200, 0),
            pygame.Rect(self.goal_state[0] - self.goal_tol, self.goal_state[1] - self.goal_tol, 20, 20),
        )

        pygame.draw.circle(
            surface,
            (0, 0, 255),
            np.array([self.state[0], self.state[2]]).astype(int),
            10
        )

        for x_start, y_start, x_end, y_end in self.walls:
            pygame.draw.line(
                surface,
                (0, 0, 0),
                (x_start, y_start),
                (x_end, y_end),
                5
            )

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(surface).swapaxes(0, 1)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def get_random_start_position(self):
        while True:
            random_x = np.random.uniform(10, self.size - 10)
            random_y = np.random.uniform(10, self.size - 10)
            proposed_start = np.array([random_x, random_y], dtype=np.float32)

            if self.is_position_legal(proposed_start):
                return proposed_start

    def normalize_obs(self, obs):
        # Normalize x and y to [-1, 1]
        x_norm = (obs[0] - self.size / 2) / (self.size / 2)
        y_norm = (obs[2] - self.size / 2) / (self.size / 2)

        # Normalize vx and vy by expected max speed (e.g. 25 px/step)
        vx_norm = obs[1] / 25.0
        vy_norm = obs[3] / 25.0

        return np.array([x_norm, vx_norm, y_norm, vy_norm], dtype=np.float32)

    @staticmethod
    def denormalize_action(action):
        return action * 10

    def is_position_legal(self, position):
        x, y = position
        for x_start, y_start, x_end, y_end in self.walls:
            if x_start == x_end:
                if (x_start - 5 <= x <= x_start + 5) and (y_start <= y <= y_end):
                    return False
            if y_start == y_end:
                if (y_start - 5 <= y <= y_start + 5) and (x_start <= x <= x_end):
                    return False
        return True


