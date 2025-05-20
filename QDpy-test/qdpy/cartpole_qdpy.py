#!/usr/bin/env python3
"""
QDpy: MAP-Elites + CartPole (OpenAI Gym)  
"""

import gym, numpy as np
from functools import partial
from qdpy import algorithms, containers, plots

#hiperparametry
ENV_NAME, DIMENSION = "CartPole-v1", 4 #środowisko
BUDGET, BATCH_SIZE   = 25_000, 512 #maksymalna liczba ewaluacji i batch osobników
GRID_SHAPE           = (64, 64) #rozmiar mapy
WEIGHT_RANGE         = (-10.0, 10.0)   
FEATURE_DOMAIN       = ((-1.0, 1.0), (-1.0, 1.0)) #zakres cech 
FITNESS_DOMAIN       = ((0.0, 500.0),)          

#polityka liniowa
def linear_policy(theta, obs):                  
    return int(np.dot(theta, obs) > 0.0)

#ocena
def evaluate_cartpole(theta, n_episodes=1):
    env = gym.make(ENV_NAME)                   
    reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(linear_policy(theta, obs)) #stan, nagroda, warunki zakonczenia
            done, reward = term or trunc, reward + r
    env.close()
    return reward / n_episodes


# ocena
def qdpy_eval(ind, n_episodes=1):
    theta = np.asarray(ind, dtype=float)
    reward = evaluate_cartpole(theta, n_episodes)

    #deskryptory - mapa
    b1, b2 = np.clip(theta[:2] / WEIGHT_RANGE[1], -1.0, 1.0)
    return (reward,), (b1, b2)                 

#kontener siatka dla MAP Elites
grid = containers.Grid(shape=GRID_SHAPE, #zakres (64x64)
                       max_items_per_bin=1, #jedno rozwiązanie
                       fitness_domain=FITNESS_DOMAIN,
                       features_domain=FEATURE_DOMAIN)


#losowe wyszukiwanie z mutacjami
algo = algorithms.RandomSearchMutPolyBounded(   
        grid,
        budget=BUDGET, #25 000
        batch_size=BATCH_SIZE, #512
        dimension=DIMENSION,
        bounds=[WEIGHT_RANGE]*DIMENSION,        
        optimisation_task="maximisation")

logger = algorithms.TQDMAlgorithmLogger(algo) #pasek

if __name__ == "__main__":
    best = algo.optimise(partial(qdpy_eval, n_episodes=3))
    print("\n PODSUMOWANIE \n", algo.summary())
    plots.default_plots_grid(logger)
    print(f"Wyniki zapisane w pliku: {logger.final_filename}")
