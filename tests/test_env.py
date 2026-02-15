import pandas as pd
from envs.trading_env import TradingEnv


def test_env_initialization():
    df = pd.read_csv("data/raw/AAPL.csv")
    env = TradingEnv(df)
    obs, _ = env.reset()
    assert len(obs) == 5

