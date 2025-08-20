
**AM-IHF** is an open-source framework designed for adaptive task allocation in **spatial crowdsourcing (SC)**. It integrates **reinforcement learning (RL)**, **Carla-based simulation**, and the **CrowHITL system** with human-in-the-loop mechanisms.


## 📂 Project Structure

- `RL/`  
  Contains RL algorithms, environments, and training scripts (e.g., `run_XXX.py`).
- `carla/`  
  Configuration files and driving algorithms for the **Carla simulator**.
- `system/`  
  Implementation of the **CrowHITL system** (visualization, HITL modules, configs).
- `util/`  
  Utility functions for data preprocessing and computational tools.
- `enable_sim_env.py`  
  Script for launching the Carla simulation environment.


## 🔧 Requirements

Make sure you are using **Python 3.7** and install the following dependencies:

```bash
carla==0.9.15
networkx==2.6.3
torch>=1.13.1
dash>=1.0.2
dash-bootstrap-components==1.2.1
dash-daq==0.1.7
numpy>=1.16.2
pandas>=0.24.2
```

## 🚀 Getting Started

Follow these steps to launch the simulation and train RL models with CrowHITL.

### 1️⃣ Start the Carla Simulation

Run the following command:

```bash
python enable_sim_env.py
```

Wait until the Carla simulation interface appears.

### 2️⃣ Launch RL Training and CrowHITL

Navigate to the `RL/` folder and select the RL algorithm you want to run. For example:

```bash
python run_DQN.py
```

This will start the CrowHITL system and connect it to the selected RL model, enabling human-in-the-loop guidance during training.

### 3️⃣ Access the CrowHITL Interface

Open a web browser and go to:

```bash
http://127.0.0.1:8050/
```

You can now interact with the system to monitor RL training, visualize worker and task distributions, and provide implicit human feedback.

