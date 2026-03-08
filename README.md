# 📈 Deep Reinforcement Learning Trading Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-00d1b2)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-FF6F00)](https://stable-baselines3.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)

An end-to-end Reinforcement Learning (RL) pipeline designed for automated financial decision-making. This project features a custom-built market simulator and a **Proximal Policy Optimization (PPO)** agent trained to maximize risk-adjusted returns through continuous interaction with historical OHLCV data.

---

## 🧠 System Design & Logic

This framework demonstrates a rigorous engineering approach to applying RL to non-stationary financial time series:

### 1. Custom Gymnasium Environment
I developed `TradingEnv`, a custom environment that simulates a brokerage account. It includes stateful balance management, share tracking, and a **penalized reward function** designed to discourage excessive drawdowns and promote capital preservation.



### 2. State Space Engineering
The observation space consists of normalized **OHLCV** (Open, High, Low, Close, Volume) data. By normalizing these inputs, the agent can perceive price momentum and volatility patterns independently of absolute price scales, improving generalization across different tickers.

### 3. PPO Algorithm
The project utilizes **Proximal Policy Optimization (PPO)** for its superior stability. Financial environments are notoriously "noisy"; PPO's clipped objective function prevents the policy from making catastrophic updates based on market outliers.

---

## 🛠 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **RL Framework** | Stable-Baselines3 | Production-grade PPO and policy network implementation. |
| **Simulation** | Gymnasium | Standardized interface for agent-environment interaction. |
| **Inference API** | FastAPI | High-speed prediction service for real-time trade signals. |
| **Data Handling** | Pandas / NumPy | Vectorized preprocessing of market datasets. |
| **CI/CD** | GitHub Actions | Automated linting and environment API validation. |

---

## 🚀 Engineering Highlights

* **Robust Environment Validation:** Before training, the system executes `check_env(env)` to ensure the Gymnasium implementation adheres to API specifications, preventing silent bugs in observation or action spaces.
* **Production Inference Service:** The model is wrapped in a FastAPI service (`serve.py`) that accepts raw market states and returns a deterministic action (Buy/Sell/Hold) with **sub-10ms latency**.
* **Automated Testing:** Includes unit tests for environment initialization, ensuring that observation shapes and initial portfolio balances remain consistent across diverse datasets.

---

## 📊 Evaluation & Strategy Performance

Training progress is monitored via TensorBoard, focusing on `explained_variance` and `approx_kl`. 

> **Metric Focus:** While raw profit is the goal, the reward function is tuned for the **Sharpe Ratio**, rewarding the agent for achieving returns with lower volatility rather than simply taking high-leverage gambles.

---

## 🏁 Quick Start

### 1. Installation
```bash
git clone [https://github.com/your-username/rl-trading-framework.git](https://github.com/your-username/rl-trading-framework.git)
cd rl-trading-framework
pip install -r requirements.txt
```
### 2. Training the Agent

Train the PPO agent on historical data (e.g., AAPL):
```bash
python train.py
```
### 3. Evaluation & Backtesting

Run the agent through a test episode to visualize the final net worth and trade execution:
```bash
python evaluate.py
```
### 4. Deploying the Signal API

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```
---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dice Score | ~0.78 (example) |
   
---

## 📉 Future Clinical Roadmap

* **[ ] Attention U-Net:** Implement Attention Gates to focus the model on small, sub-centimeter nodules.
* **[ ] Malignancy Classifier:** Use extracted Radiomics features to train a Random Forest classifier.
* **[ ] 3D Slicer Plugin:** Integrate the model directly into clinical viewing software.

> **Disclaimer:** This tool is for research purposes only and is not cleared for clinical diagnosis.
