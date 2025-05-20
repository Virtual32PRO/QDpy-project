import pickle, gym
import imageio.v2 as imageio
from cartpole_qdpy import linear_policy
import os

# Wczytaj najlepszego osobnika
with open("final.p", "rb") as f:
    data = pickle.load(f)
    best = data["container"].best

# Uruchom środowisko w trybie obrazu
env = gym.make("CartPole-v1", render_mode="rgb_array")
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

output_path = os.path.join(os.path.dirname(__file__), "cartpole_best.gif")

print(f"GIF zapisany jako: {output_path}")

# ✅ Zapis do GIF-a — nie potrzebujesz ffmpeg!
imageio.mimsave(output_path, frames, fps=30)
print(" GIF zapisany jako 'cartpole_best.gif'")
