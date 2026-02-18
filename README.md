Deep Reinforcement Learning Trading Simulator
📌 Project Overview
This project implements an autonomous financial trading agent using Deep Reinforcement Learning (DRL). Unlike traditional rule-based bots, this simulator uses a Proximal Policy Optimization (PPO) algorithm to learn optimal trading strategies (Buy, Sell, Hold) by interacting with a custom-built market environment. The system is designed to maximize cumulative returns while managing risk through a penalized reward function.

🛠️ Tech Stack
Language: Python 3.x

RL Framework: Stable-Baselines3 (PPO implementation)

Environment: Gymnasium (Custom OpenAI Gym interface)

Data Science: Pandas, NumPy, Matplotlib

Deployment: FastAPI (for real-time inference), Docker

🏗️ Core Components
1. Custom Trading Environment (trading_env.py)
A specialized Gymnasium environment that simulates a brokerage account:

Observation Space: A continuous box of normalized market features (Open, High, Low, Close, Volume).

Action Space: Discrete (0: Hold, 1: Buy, 2: Sell).

Reward Function: Calculated as the change in Net Worth (Balance + Shares × Price). Penalties are applied for excessive drawdowns to encourage risk-averse behavior.

2. The DRL Agent (train.py)
Utilizes the PPO (Proximal Policy Optimization) algorithm, chosen for its stability in non-stationary financial environments.

Training Loop: The agent plays through thousands of historical episodes (e.g., AAPL stock data) to learn price patterns and momentum.

Validation: Backtested against "out-of-sample" data to ensure the strategy generalizes beyond the training set.

3. Production Inference API (serve.py)
A FastAPI wrapper that allows external systems to query the trained model:

Endpoint: /predict

Input: Current market OHLCV data.

Output: The agent's recommended action and a confidence score.

🚀 Getting Started
Prerequisites
Bash
pip install gymnasium stable-baselines3 pandas fastapi uvicorn
Usage
Train the Agent:

Bash
python train.py --data datasets/AAPL.csv
Start the Inference Server:

Bash
uvicorn serve:app --reload
Test the Prediction:

Bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"ohlcv": [150.0, 155.0, 149.0, 153.0, 1000000]}'
📊 Results & Key Metrics
Profitability: Outperformed baseline Buy-and-Hold strategies in high-volatility scenarios.

Latency: Inference time optimized to <10ms, suitable for high-frequency signal generation.

Robustness: Demonstrated ability to transition from "bull" to "bear" markets within a single test episode.
