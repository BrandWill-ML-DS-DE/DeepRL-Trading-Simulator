import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        # Action: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation: Open, High, Low, Close, Volume (Normalized)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.df.iloc[self.current_step][["Open", "High", "Low", "Close", "Volume"]]
        return obs.values.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]
        
        if action == 1 and self.balance >= current_price: # Buy
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0: # Sell
            self.shares_held -= 1
            self.balance += current_price

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        net_worth = self.balance + (self.shares_held * current_price)
        reward = (net_worth - self.initial_balance) / self.initial_balance # Percent return
        
        return self._get_obs(), reward, done, False, {"net_worth": net_worth}
