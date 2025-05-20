import pickle, gym
import imageio.v2 as imageio
import numpy as np
import os

from lunarlander_qdpy import linear_policy  # <-- zaimportuj odpowiednią politykę

# Wczytaj najlepszego osobnika
with open("final.p", "rb") as f:
    data = pickle.load(f)
    best = data["container"].best

# Uruchom środowisko z wizualizacją
env = gym.make("LunarLander-v2", render_mode="rgb_array")
obs, _ = env.reset()
done = False
frames = []

while not done:
    frame = env.render()
    frames.append(frame)
    action = linear_policy(best, obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()

# Zapisz GIF do bieżącego folderu
output_path = os.path.join(os.path.dirname(__file__), "lunarlander_best.gif")
imageio.mimsave(output_path, frames, fps=30)
print(f"✅ GIF zapisany jako: {output_path}")