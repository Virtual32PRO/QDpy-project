#!/usr/bin/env python3
"""
MAP-Elites + LunarLander-v2 (OpenAI Gym)
"""
import os
os.chdir("C:/Users/jacek/OneDrive/Pulpit/qdpy_1")
print("Bieżący folder:", os.getcwd())


import gym
import numpy as np
from functools import partial
from qdpy import algorithms, containers, plots

#hiperparametry
ENV_NAME          = "LunarLander-v2"   
OBS_DIM           = 8                  
ACT_DIM           = 4
DIMENSION         = OBS_DIM * ACT_DIM + ACT_DIM   # parametry w genotypie
BUDGET            = 50000             # liczba ewaluacji
BATCH_SIZE        = 1024
GRID_SHAPE        = (64, 64)           
WEIGHT_RANGE      = (-5.0, 5.0)        # inicjalizacja + mutacje
FEATURE_DOMAIN    = ((0.0, 1.0), (0.0, 1.0))  
FITNESS_DOMAIN    = ((-300.0, 300.0),)        

#polityka
def linear_policy(theta: np.ndarray, obs: np.ndarray) -> int:
    theta = np.asarray(theta)
    W = theta[:OBS_DIM * ACT_DIM].reshape(OBS_DIM, ACT_DIM)   
    b = theta[-ACT_DIM:]                                      
    logits = obs @ W + b
    return int(np.argmax(logits))                             

# ---------- EWALUACJA ----------
def evaluate_lunar(theta: np.ndarray,
                   n_episodes: int = 1,
                   render: bool = False):
    env = gym.make(ENV_NAME, render_mode="human" if render else None)
    total_reward, feat_x, feat_ang = 0.0, 0.0, 0.0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = linear_policy(theta, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        x_pos   = abs(obs[0])           
        angle   = abs(obs[4])         
        feat_x  += min(x_pos / 1.5, 1.0)
        feat_ang += min(angle / np.pi, 1.0)

    env.close()
    mean_reward = total_reward / n_episodes
    mean_x      = feat_x / n_episodes
    mean_ang    = feat_ang / n_episodes
    return mean_reward, mean_x, mean_ang

#ocena
def qdpy_eval(ind, n_episodes=1):
    theta = np.asarray(ind, dtype=float)
    reward, fx, fang = evaluate_lunar(theta, n_episodes=n_episodes)
    return (reward,), (fx, fang)        

#kontener siatka dla MAP-Elites
grid = containers.Grid(shape=GRID_SHAPE,
                       max_items_per_bin=1,
                       fitness_domain=FITNESS_DOMAIN,
                       features_domain=FEATURE_DOMAIN)


#losowe wyszukiwanie z mutacjami
algo = algorithms.RandomSearchMutPolyBounded(
        grid,
        budget=BUDGET,
        batch_size=BATCH_SIZE,
        dimension=DIMENSION,
        bounds=[WEIGHT_RANGE] * DIMENSION,
        optimisation_task="maximisation")

logger = algorithms.TQDMAlgorithmLogger(algo) #pasek



if __name__ == "__main__":
    best = algo.optimise(partial(qdpy_eval, n_episodes=2))

    print(" \PODSUMOWANIE MAP-ELITES ")
    print(algo.summary())

    plots.default_plots_grid(logger)       
    print(f"Wyniki zapisane w: {logger.final_filename}")
