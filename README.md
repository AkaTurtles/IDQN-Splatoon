# IDQN-Splatoon
An Multiagent Reinforcement Learning IDQN agent that plays Splatoon

This project implements a Splatoon-inspired multi-agent reinforcement learning environment where two teams of agents compete to cover a grid with their team colour.
Agents are trained with Independent Deep Q-Networks (IDQN) and evaluated against a rule-based heuristic.
You can then watch trained agents play live in a Pygame viewer.

Requirements
Python 3.10+

Recommended packages:
numpy
torch
gymnasium
matplotlib
pygame

Install everything (adjust as needed):
pip install numpy torch gymnasium matplotlib pygame

Key files:
splatoon.py
Gymnasium environment splatoon implementing the grid-based Splatoon-like game (movement, painting, splatting, rewards, rendering).

idqn.py (or idqn.py depending on your filename)
IDQN implementation:
extract_local_obs – builds per-agent local observations.
TeamIDQN – shared Q-network per team.
train_idqn – training loop (experience replay, target network, epsilon-greedy, learning curve plotting).

testbed.py
Main training & evaluation script.
Trains IDQN teams in the Splatoon environment and usually:
Saves checkpoints into checkpoints/.
Runs evaluation games vs a heuristic agent.
Produces plots comparing performance (coverage, returns, win-rate).

demo_live.py
Live demo / viewer.
Loads saved IDQN checkpoints and plays a few episodes of Splatoon in a Pygame window so you can watch the learned behaviour.

checkpoints/
Folder where trained models are saved:
team0_idqn.pt
team1_idqn.pt

Make sure the checkpoints folder exists in the repo root:
mkdir -p checkpoints

3. Training & Generating Models (using testbed.py)
To train the IDQN agents and generate models:
python testbed.py

What this script typically does (high-level):
Create the environment
env = SplatoonGymEnv(H=9, W=9, team_sizes=(2, 2), max_steps=200, render_mode=None)

Train IDQN teams
team0, team1 = train_idqn(
    env,
    total_steps=80_000,        # or whatever you configured
    device="cpu",
    obs_patch=7,
    log_every=2000,
    seed=0,
    plot_learning_curve=True   # will show/save learning curve
)

Save checkpoints
Inside train_idqn or testbed.py, models are saved to:
team0.save("checkpoints/team0_idqn.pt")
team1.save("checkpoints/team1_idqn.pt")

Run testbed.py until you are happy with the learning curve and the saved models in checkpoints/.

4. Running the Live Demo (demo_live.py)
Once you have trained models and .pt files in checkpoints/, you can watch them play:
python demo_live.py

demo_live.py does roughly:
a) Create a rendered environment
env = SplatoonGymEnv(
    H=9, W=9,
    team_sizes=(2, 2),
    max_steps=120,
    render_mode="human"   # enables Pygame rendering
)

b) Build IDQN teams and load checkpoints
sample_vec = extract_local_obs(obs, 0, k=7)
obs_dim = sample_vec.shape[0]
cfg = IDQNConfig(obs_patch=7)
team0 = TeamIDQN(obs_dim, cfg.n_actions, cfg)
team1 = TeamIDQN(obs_dim, cfg.n_actions, cfg)
team0.load("checkpoints/team0_idqn.pt")
team1.load("checkpoints/team1_idqn.pt")
If your paths differ, edit the load(...) calls in demo_live.py to point to your actual files.

c) Control teams during the demo
Typically: Team 0 uses team0.act_greedy(...)
Team 1 can be:
Another IDQN (team1.act_greedy(...)), or
The heuristic agent, via HeuristicController, depending on how you set the logic in the script.

d) Render a few episodes
play_live() loops over several episodes and calls env.step(actions) + env.render() with a small time.sleep(...) so spectators can watch.


6. Quick Start Summary
Train & generate models
python testbed.py
# -> checkpoints/team0_idqn.pt, checkpoints/team1_idqn.pt

(Optional) Run quantitative comparison
python idqn_vs_heuristic.py

Run live visual demo
python demo_live.py

https://youtu.be/GBP_6Dtk4Lw
