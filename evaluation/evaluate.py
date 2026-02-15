import pandas as pd
from stable_baselines3 import PPO
from envs.trading_env import TradingEnv


def evaluate():
    df = pd.read_csv("data/raw/AAPL.csv")
    env = TradingEnv(df)

    model = PPO.load("agents/ppo_trading")

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

    print("Final Net Worth:", env.net_worth)


if __name__ == "__main__":
    evaluate()

