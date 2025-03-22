import gymnasium as gym
import numpy as np
from gymnasium import register

env_name = "CustomPointMassEnv"
env_id = "CustomPointMass-v0"

register(
        id=f"{env_id}",
        entry_point=f"envs.{env_name}:{env_name}",
       # kwargs={"max_steps": 500, "fixed_start": False},  # ✅ Set max steps here

    )

env = gym.make(env_id, render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, _, info = env.step(np.array([1.0, 1.0]))  # constant force
    env.render()  # Show visualization with MushroomRL

env.close()