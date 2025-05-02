# navigation-rl-agent

Hello! I’m **Kashad J. Turner-Warren**, an undergraduate Mathematics student at Arizona State University and a passionate AI & cybersecurity practitioner. In this project, I designed and trained a Deep Q-Network (DQN) to solve Unity’s Banana Collector environment, demonstrating my skills in reinforcement learning, software engineering, and analytical problem-solving.

---

## 🍌 Environment Details

The agent must navigate a large, square world and collect bananas while avoiding pitfalls:

* **Rewards**: +1 for collecting a **yellow** banana; –1 for collecting a **blue** banana.
* **State space (37 dimensions)**: Includes the agent’s velocity vector and ray-based distance measurements to objects in front of it.
* **Action space (4 discrete actions)**:

  1. Move forward
  2. Move backward
  3. Turn left
  4. Turn right
* **Task type**: Episodic – each episode ends after a fixed number of steps or when manual termination.
* **Solved criterion**: Achieve an average score of **+13** over 100 consecutive episodes.

By framing this as a classic exploration–exploitation challenge, I applied DQN to learn optimal navigation policies in a sparse-reward setting.

---

## 🎯 Why This Project Matters

* **Technical Rigor**: I integrated core DQN innovations—experience replay, target networks, and ε–greedy exploration—following Mnih et al. (2015) and Udacity’s RL curriculum.
* **Production-Ready Engineering**: I built a modular codebase (`src/`) with an environment wrapper, training CLI, and well-structured agent classes.
* **Demonstrated Impact**: My agent reached the +13 average reward threshold in **\~600 episodes**, validating stability and efficiency.

These outcomes underscore my capability to turn academic research into robust, real-world solutions—an asset for roles in AI, robotics, or ML infrastructure.

---

## 🛠️ Implementation Highlights

1. **DQN Mastery**

   * Implemented a PyTorch Q-network (2 hidden layers) with ReLU activations.
   * Designed a replay buffer storing up to 100k transitions for decorrelated updates.
   * Tuned hyperparameters (learning rate, batch size, discount factor, ε-decay) through iterative experiments.

2. **Environment Integration**

   * Created `src/environment_wrapper.py` to translate Unity’s ML-Agents API into a Gym-like interface.
   * Parsed 37-dimensional observations and mapped discrete action indices to Unity agent commands.

3. **Training Pipeline**

   * Developed `src/train.py` for training and evaluation, with CLI flags (`--train`, `--eval`, `--env-path`, etc.).
   * Logged performance metrics and saved checkpoints (`models/checkpoint_final.pt`).
   * Generated reward curves (`results/rewards_plot.png`) for visual analysis.

4. **Quality Assurance**

   * Wrote unit tests (`tests/`) for action selection, replay buffer sampling, and environment resets.
   * Automated sanity checks in CI-style runs to catch regressions early.

---

## 🍌 Getting the Environment

Before running the code, you need to download the Unity Banana Collector environment:

1. Download the environment that matches your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Unzip the file in a location of your choice and note the path to the executable:
   - Linux: `Banana_Linux/Banana.x86_64`
   - Mac OSX: `Banana.app`
   - Windows: `Banana_Windows_x86/Banana.exe` or `Banana_Windows_x86_64/Banana.exe`

This path will be used with the `--env-path` parameter when running the training script.

## 🚀 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/krillavilla/navigation-rl-agent.git
cd navigation-rl-agent

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Train (CPU-only by default)
python src/train.py --env-path /path/to/Banana-Collector-Env --train --episodes 2000 --batch-size 64 --lr 5e-4 --gamma 0.99

# Use GPU if available
python src/train.py --env-path /path/to/Banana-Collector-Env --train --gpu

# Evaluate saved model
echo "Evaluating trained agent..."
python src/train.py --env-path /path/to/Banana-Collector-Env --eval --load-path models/checkpoint_final.pt
```

Explore `notebooks/` for interactive walkthroughs (`Basics.ipynb`, `Navigation.ipynb`).

## 🧪 Local Testing

Follow these exact steps to test the repository locally:

```bash
# 1. Create & activate virtualenv
python3 -m venv .venv
source .venv/bin/activate        # or .\.venv\Scripts\Activate.ps1 on Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Smoke-test training for 20 episodes
python src/train.py \
  --env-path environment/Banana_Linux/Banana.x86_64 \
  --train \
  --episodes 20 \
  --batch-size 32

# 4. Evaluate the saved model
python src/train.py \
  --env-path environment/Banana_Linux/Banana.x86_64 \
  --eval \
  --load-path models/checkpoint_final.pt
```

The human-agent interface with WASD controls is available in `notebooks/Navigation.ipynb` for hands-on exploration of the environment.

---

## 📈 Results & Impact

* **Solved at episode**: \~600 (100-episode average ≥ +13)
* **Peak evaluation score**: \~+15 average reward

See `results/rewards_plot.png` for the training curve.

---

## 📄 Report & Next Steps

Review **`docs/Report.md`** for full details on:

* Algorithm design and network architecture
* Hyperparameter exploration and tuning
* Learning curve analysis and solved episode annotation
* Future improvements: Double DQN, Prioritized Replay, Dueling Architecture

---

Thank you for exploring my work! I’m excited to leverage these skills in AI-driven robotics, autonomous navigation, or secure ML operations. Let’s connect!
