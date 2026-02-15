from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO

app = FastAPI(title="RL Trading API")
model = PPO.load("agents/ppo_trading")

class MarketState(BaseModel):
    values: list[float] # Expecting [Open, High, Low, Close, Volume]

@app.get("/health")
def health():
    return {"status": "ready"}

@app.post("/predict")
def predict(state: MarketState):
    if len(state.values) != 5:
        raise HTTPException(status_code=400, detail="Invalid input length")
    
    obs = np.array(state.values, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    
    actions_map = {0: "Hold", 1: "Buy", 2: "Sell"}
    return {"action_code": int(action), "action_name": actions_map[int(action)]}
