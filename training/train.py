import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.trading_env import TradingEnv


def train():
    df = pd.read_csv("data/raw/AAPL.csv")
    env = TradingEnv(df)

    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)

    model.save("agents/ppo_trading")


if __name__ == "__main__":
    train()

