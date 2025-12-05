# compare_idqn_vs_heuristic.py

import numpy as np
import matplotlib.pyplot as plt

from splatoon import SplatoonGymEnv
from idqn import TeamIDQN, IDQNConfig, extract_local_obs

# Action indices must match your env
STAY, UP, DOWN, LEFT, RIGHT, SPRAY, SPLAT = range(7)
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

class HeuristicController:
    """
    Simple rule-based controller:
    - Splat if enemy adjacent
    - Else move onto adjacent neutral/enemy tile
    - Else drift towards enemy side
    """
    def __init__(self, grid_h, grid_w):
        self.H = grid_h
        self.W = grid_w

    def act_for_agent(self, obs, agent_idx):
        grid = obs["grid"]          # (H, W) in {-1,0,1}
        positions = obs["positions"]  # (n,2)
        teams = obs["teams"]
        r, c = positions[agent_idx]
        r, c = int(r), int(c)
        my_team = int(teams[agent_idx])

        # 1) Check for adjacent enemies -> SPLAT
        for i, (rr, cc) in enumerate(positions):
            if i == agent_idx:
                continue
            if int(teams[i]) == my_team:
                continue
            rr, cc = int(rr), int(cc)
            if abs(rr - r) + abs(cc - c) == 1:
                return SPLAT

        # 2) Paint nearby neutral/enemy tiles by moving into them
        best_move = None
        best_score = -1
        for a, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W:
                tile = int(grid[nr, nc])
                # prefer neutral > enemy > own
                score = 0
                if tile == -1:         # neutral
                    score = 2
                elif tile != my_team:  # enemy
                    score = 1
                if score > best_score:
                    best_score = score
                    best_move = a

        if best_move is not None and best_score > 0:
            return best_move

        # 3) Drift towards enemy side of the map
        # team 0 -> move right; team 1 -> move left
        if my_team == 0 and c < self.W - 1:
            return RIGHT
        if my_team == 1 and c > 0:
            return LEFT

        # 4) Default: stay
        return STAY


def load_idqn_teams(env, ckpt0_path, ckpt1_path, obs_patch=7, device="cpu"):
    """Construct TeamIDQN objects and load checkpoints."""
    obs, info = env.reset(seed=123)
    sample_vec = extract_local_obs(obs, agent_idx=0, k=obs_patch)
    obs_dim = sample_vec.shape[0]

    cfg = IDQNConfig(obs_patch=obs_patch)
    team0 = TeamIDQN(obs_dim, cfg.n_actions, cfg, device=device)
    team1 = TeamIDQN(obs_dim, cfg.n_actions, cfg, device=device)
    team0.load(ckpt0_path, map_location=device)
    team1.load(ckpt1_path, map_location=device)
    return team0, team1


def run_eval(env, controller_A, controller_B, teams, n_episodes=50,
             idqn_A=None, idqn_B=None, obs_patch=7, seed=0):
    """
    controller_A / controller_B:
        either "heuristic" or "idqn"
    idqn_A / idqn_B:
        TeamIDQN instances when using IDQN
    teams:
        obs["teams"] array from env (which agents belong to which team)
    """

    rng = np.random.default_rng(seed)

    per_ep_cov_A = []
    per_ep_cov_B = []
    per_ep_rew_A = []
    per_ep_rew_B = []
    per_ep_winner = []  # 0,1, or -1 for draw

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 10_000)))
        coverage = info["coverage"]  # [covA, covB]
        n_agents = len(obs["positions"])
        teams_arr = obs["teams"]
        ep_rew_A = 0.0
        ep_rew_B = 0.0
        done = False

        while not done:
            actions = np.zeros(n_agents, dtype=np.int64)
            for i in range(n_agents):
                team_i = int(teams_arr[i])

                if team_i == 0:  # Team A
                    if controller_A == "heuristic":
                        h = HeuristicController(env.H, env.W)
                        actions[i] = h.act_for_agent(obs, i)
                    else:  # "idqn"
                        vec = extract_local_obs(obs, i, k=obs_patch)
                        actions[i] = idqn_A.act_greedy(vec)
                else:  # Team B
                    if controller_B == "heuristic":
                        h = HeuristicController(env.H, env.W)
                        actions[i] = h.act_for_agent(obs, i)
                    else:  # "idqn"
                        vec = extract_local_obs(obs, i, k=obs_patch)
                        actions[i] = idqn_B.act_greedy(vec)

            obs, rew_vec, term, trunc, info = env.step(actions)
            done = bool(term or trunc)
            ep_rew_A += rew_vec[teams_arr == 0].sum()
            ep_rew_B += rew_vec[teams_arr == 1].sum()
            coverage = info["coverage"]

        per_ep_cov_A.append(coverage[0])
        per_ep_cov_B.append(coverage[1])
        per_ep_rew_A.append(ep_rew_A)
        per_ep_rew_B.append(ep_rew_B)

        if coverage[0] > coverage[1]:
            per_ep_winner.append(0)
        elif coverage[1] > coverage[0]:
            per_ep_winner.append(1)
        else:
            per_ep_winner.append(-1)

    results = {
        "cov_A": np.array(per_ep_cov_A),
        "cov_B": np.array(per_ep_cov_B),
        "rew_A": np.array(per_ep_rew_A),
        "rew_B": np.array(per_ep_rew_B),
        "winner": np.array(per_ep_winner),
    }
    return results


def summarize_results(label, res):
    cov_A, cov_B = res["cov_A"], res["cov_B"]
    rew_A, rew_B = res["rew_A"], res["rew_B"]
    winners = res["winner"]
    n = len(winners)
    win_A = np.mean(winners == 0)
    win_B = np.mean(winners == 1)
    draw = np.mean(winners == -1)
    print(f"\n=== {label} ===")
    print(f"Mean coverage A: {cov_A.mean():.3f}, B: {cov_B.mean():.3f}")
    print(f"Mean return A:   {rew_A.mean():.2f}, B: {rew_B.mean():.2f}")
    print(f"Win rate A: {win_A:.3f}, B: {win_B:.3f}, Draw: {draw:.3f}")


def plot_results(res_heur_vs_heur, res_idqn_vs_heur, out_path=None):
    # Basic plots: coverage + win rate
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    # 1) Coverage boxplots
    axs[0].boxplot(
        [res_heur_vs_heur["cov_A"], res_idqn_vs_heur["cov_A"]],
        labels=["Heur A vs Heur B", "IDQN A vs Heur B"]
    )
    axs[0].set_title("Team A Coverage")
    axs[0].set_ylabel("Coverage fraction")

    # 2) Return boxplots
    axs[1].boxplot(
        [res_heur_vs_heur["rew_A"], res_idqn_vs_heur["rew_A"]],
        labels=["Heur A", "IDQN A"]
    )
    axs[1].set_title("Team A Episode Return")

    # 3) Win rate bar chart
    def win_rate(res):
        winners = res["winner"]
        return np.mean(winners == 0)

    win_heur = win_rate(res_heur_vs_heur)
    win_idqn = win_rate(res_idqn_vs_heur)
    axs[2].bar([0, 1], [win_heur, win_idqn])
    axs[2].set_xticks([0, 1])
    axs[2].set_xticklabels(["Heur A vs Heur B", "IDQN A vs Heur B"])
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel("Win rate (Team A)")
    axs[2].set_title("Win Rate Comparison")

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # 1) Create env
    env = SplatoonGymEnv(H=9, W=9, team_sizes=(2, 2), max_steps=200, render_mode=None)
    obs, info = env.reset(seed=0)
    teams_arr = obs["teams"]

    # 2) Load trained IDQN models
    team0, team1 = load_idqn_teams(
        env,
        ckpt0_path="D:/University/COMP4900 multiagent reinforcement learning/checkpoints/team0_idqn.pt",
        ckpt1_path="D:/University/COMP4900 multiagent reinforcement learning/checkpoints/team1_idqn.pt",
        obs_patch=7,
        device="cpu",
    )

    # 3) Evaluate Heuristic vs Heuristic (baseline)
    res_hvh = run_eval(
        env,
        controller_A="heuristic",
        controller_B="heuristic",
        teams=teams_arr,
        n_episodes=50,
        idqn_A=None,
        idqn_B=None,
        obs_patch=7,
    )

    # 4) Evaluate IDQN (Team A) vs Heuristic (Team B)
    res_ivh = run_eval(
        env,
        controller_A="idqn",
        controller_B="heuristic",
        teams=teams_arr,
        n_episodes=50,
        idqn_A=team0,
        idqn_B=None,      # Team B uses heuristic
        obs_patch=7,
    )

    summarize_results("Heuristic vs Heuristic", res_hvh)
    summarize_results("IDQN vs Heuristic", res_ivh)

    plot_results(res_hvh, res_ivh, out_path="idqn_vs_heuristic.png")
