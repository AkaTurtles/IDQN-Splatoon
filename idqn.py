# idqn_splatoon.py
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict, Tuple, List
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =========================
#  Utils: local observations
# =========================

def extract_local_obs(obs: Dict, agent_idx: int, k: int = 7) -> np.ndarray:
    """
    Build a per-agent observation vector:
      - local k x k crop around the agent from obs["grid"] (values in {-1,0,1})
      - agent team (0/1), stunned (0/1)
      - normalized time t / t_max
    Returns float32 flat vector.
    """
    grid = obs["grid"]         # (H,W) int8 in {-1,0,1}
    pos = obs["positions"]     # (n,2)
    stunned = obs["stunned"]   # (n,)
    teams = obs["teams"]       # (n,)
    t = obs["t"]

    H, W = grid.shape
    r, c = pos[agent_idx]
    r, c = int(r), int(c)
    half = k // 2

    # pad grid for easy crop
    pad_val = -1
    padded = np.pad(grid, ((half, half), (half, half)), mode="constant", constant_values=pad_val)
    rr, cc = r + half, c + half
    crop = padded[rr - half: rr + half + 1, cc - half: cc + half + 1]  # (k,k)

    # normalize to [-1,1] already; just cast float32
    crop = crop.astype(np.float32)

    # extra scalars
    team = float(teams[agent_idx])
    st = float(stunned[agent_idx])
    # if you know max steps from env, pass via wrapper, otherwise safe scale
    t_norm = float(t / 200.0)

    # concat
    vec = np.concatenate([crop.flatten(), np.array([team, st, t_norm], dtype=np.float32)])
    return vec

# =========================
#  Replay Buffer
# =========================

Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        # obs, next_obs: (obs_dim,) float32
        self.buf.append(Transition(obs, int(action), float(reward), next_obs, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs = torch.tensor(np.stack([b.obs for b in batch], axis=0), dtype=torch.float32)
        act = torch.tensor([b.action for b in batch], dtype=torch.long)
        rew = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        nxt = torch.tensor(np.stack([b.next_obs for b in batch], axis=0), dtype=torch.float32)
        done = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return obs, act, rew, nxt, done

    def __len__(self):
        return len(self.buf)

# =========================
#  Q-Network
# =========================

class MLPQ(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], n_actions)
        )

    def forward(self, x):
        return self.net(x)

# =========================
#  IDQN Agent (parameter-shared per team)
# =========================

@dataclass
class IDQNConfig:
    obs_patch: int = 7
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    start_learning: int = 5_000
    target_tau: float = 1.0     # hard update if 1.0
    target_update_every: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    train_freq: int = 4
    grad_clip: float = 10.0
    n_actions: int = 7

class TeamIDQN:
    """
    One DQN per team, shared across that team's agents.
    """
    def __init__(self, obs_dim: int, n_actions: int, cfg: IDQNConfig, device="cpu"):
        self.device = torch.device(device)
        self.q = MLPQ(obs_dim, n_actions).to(self.device)
        self.target = MLPQ(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.cfg = cfg
        self.steps = 0
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim)

    def epsilon(self):
        # linear decay
        frac = min(1.0, self.steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac

    @torch.no_grad
    def act(self, obs_vec: np.ndarray) -> int:
        self.q.eval()
        if random.random() < self.epsilon():
            return random.randrange(self.cfg.n_actions)
        x = torch.tensor(obs_vec[None, :], dtype=torch.float32, device=self.device)
        q = self.q(x)
        return int(q.argmax(dim=1).item())

    def push(self, *args):
        self.buffer.push(*args)

    def maybe_update(self):
        self.steps += 1
        if len(self.buffer) < self.cfg.start_learning:
            return None
        if self.steps % self.cfg.train_freq != 0:
            return None

        self.q.train()
        obs, act, rew, nxt, done = self.buffer.sample(self.cfg.batch_size)
        obs, act, rew, nxt, done = obs.to(self.device), act.to(self.device), rew.to(self.device), nxt.to(self.device), done.to(self.device)

        # Q-learning target
        with torch.no_grad():
            q_next = self.target(nxt)              # (B, A)
            a_next = q_next.max(dim=1).values      # max_a' Q_target(s',a')
            target = rew + (1.0 - done) * self.cfg.gamma * a_next

        q_pred = self.q(obs).gather(1, act.view(-1, 1)).squeeze(1)
        loss = F.smooth_l1_loss(q_pred, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        # target update
        if self.steps % self.cfg.target_update_every == 0:
            if self.cfg.target_tau >= 1.0:
                self.target.load_state_dict(self.q.state_dict())
            else:
                with torch.no_grad():
                    for p, pt in zip(self.q.parameters(), self.target.parameters()):
                        pt.data.mul_(1.0 - self.cfg.target_tau).add_(self.cfg.target_tau * p.data)

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "steps": self.steps,
            "cfg": vars(self.cfg),
        }, path)

    def load(self, path: str, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.q.load_state_dict(ckpt["q"])
        self.target.load_state_dict(ckpt["target"])
        self.opt.load_state_dict(ckpt["opt"])
        self.steps = ckpt.get("steps", 0)

    @torch.no_grad()
    def act_greedy(self, obs_vec: np.ndarray) -> int:
        self.q.eval()
        x = torch.tensor(obs_vec[None, :], dtype=torch.float32, device=self.device)
        q = self.q(x)
        return int(q.argmax(dim=1).item())


# =========================
#  Training loop
# =========================

def train_idqn(env, total_steps=300_000, device="cpu",
               obs_patch=7, log_every=2000, seed=0, plot_learning_curve=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Probe obs_dim
    obs, info = env.reset(seed=seed)
    sample_vec = extract_local_obs(obs, agent_idx=0, k=obs_patch)
    obs_dim = sample_vec.shape[0]
    n_agents = obs["positions"].shape[0]
    teams = obs["teams"]  # (n_agents,) 0/1

    cfg = IDQNConfig(obs_patch=obs_patch)
    team0 = TeamIDQN(obs_dim, cfg.n_actions, cfg, device=device)
    team1 = TeamIDQN(obs_dim, cfg.n_actions, cfg, device=device)

    # Per-agent last obs cache
    last_obs_vecs = [extract_local_obs(obs, i, obs_patch) for i in range(n_agents)]
    ep_returns = np.zeros(n_agents, dtype=np.float32)
    ep_len = 0
    global_step = 0
    episode_idx = 0

    # ---- Metrics for leaning curve ----
    episode_ids = []
    episode_mean_returns = []
    episode_cov_A = []
    episode_cov_B = []


    while global_step < total_steps:
        # Build joint action: each agent via its team network
        acts = np.zeros(n_agents, dtype=np.int64)
        for i in range(n_agents):
            if teams[i] == 0:
                acts[i] = team0.act(last_obs_vecs[i])
            else:
                acts[i] = team1.act(last_obs_vecs[i])

        next_obs, reward_vec, term, trunc, info = env.step(acts)
        done = bool(term or trunc)

        # Next obs vectors
        next_obs_vecs = [extract_local_obs(next_obs, i, obs_patch) for i in range(n_agents)]

        # Store transitions per team buffer
        for i in range(n_agents):
            if teams[i] == 0:
                team0.push(last_obs_vecs[i], acts[i], reward_vec[i], next_obs_vecs[i], done)
            else:
                team1.push(last_obs_vecs[i], acts[i], reward_vec[i], next_obs_vecs[i], done)

        # Optimize
        loss0 = team0.maybe_update()
        loss1 = team1.maybe_update()

        ep_returns += reward_vec
        ep_len += 1
        global_step += 1
        last_obs_vecs = next_obs_vecs

        if done:
            episode_idx += 1
            mean_ret = float(ep_returns.mean())
            cov = info.get("coverage", np.array([0.0, 0.0]))
            covA, covB = float(cov[0]), float(cov[1])

            # ---- Log Metrics for learning curve ----
            episode_ids.append(episode_idx)
            episode_mean_returns.append(mean_ret)
            episode_cov_A.append(covA)
            episode_cov_B.append(covB)

            #console logging
            if (global_step // log_every) >= 0 and global_step % log_every < 5:
                print(f"[step {global_step}] ep_len={ep_len} R_mean={ep_returns.mean():.3f} "
                      f"A_cov={cov[0]*100:.1f}% B_cov={cov[1]*100:.1f}% "
                      f"eps0={team0.epsilon():.2f} eps1={team1.epsilon():.2f} "
                      f"loss0={loss0 if loss0 is not None else '-'} loss1={loss1 if loss1 is not None else '-'}")
            
            #logging
            if global_step % 50_000 == 0 and global_step > 0:
                team0.save(f"checkpoints/team0_step{global_step}.pt")
                team1.save(f"checkpoints/team1_step{global_step}.pt")

            obs, info = env.reset()
            last_obs_vecs = [extract_local_obs(obs, i, obs_patch) for i in range(n_agents)]
            ep_returns[:] = 0.0
            ep_len = 0

    print("Training completed.")
    # save models
    team0.save("checkpoints/team0_idqn.pt")
    team1.save("checkpoints/team1_idqn.pt")

    # ---- Plot learning curve ----
    if plot_learning_curve and len(episode_ids) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(episode_ids, episode_mean_returns, label="Mean episode return")
        plt.xlabel("Episode")
        plt.ylabel("Mean return (per-agent)")
        plt.title("IDQN Learning Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Optional: coverage curves
        plt.figure(figsize=(8, 4))
        plt.plot(episode_ids, np.array(episode_cov_A) * 100.0, label="Team A coverage %")
        plt.plot(episode_ids, np.array(episode_cov_B) * 100.0, label="Team B coverage %")
        plt.xlabel("Episode")
        plt.ylabel("Coverage (%)")
        plt.title("Coverage over Episodes")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # return models + raw metrics in case you want to save/plot later
    metrics = {
        "episode": np.array(episode_ids),
        "mean_return": np.array(episode_mean_returns),
        "cov_A": np.array(episode_cov_A),
        "cov_B": np.array(episode_cov_B),
    }

    return team0, team1, metrics
