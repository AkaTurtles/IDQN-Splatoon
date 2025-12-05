# demo_live.py
import time
import numpy as np
from splatoon import SplatoonGymEnv
from idqn import TeamIDQN, IDQNConfig, extract_local_obs

STAY, UP, DOWN, LEFT, RIGHT, SPRAY, SPLAT = range(7)
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

class HeuristicController:
    def __init__(self, H, W, n_agents, splat_cooldown=3):
        self.H = H
        self.W = W
        self.n_agents = n_agents
        self.splat_cooldown = splat_cooldown
        # cooldown[i] = how many steps left until agent i can SPLAT again
        self.cooldown = [0 for _ in range(n_agents)]

    def reset(self):
        """Call this at the start of each episode."""
        self.cooldown = [0 for _ in range(self.n_agents)]

    def act_for_agent(self, obs, agent_idx):
        grid = obs["grid"]
        positions = obs["positions"]
        teams = obs["teams"]

        r, c = positions[agent_idx]
        r, c = int(r), int(c)
        my_team = int(teams[agent_idx])

        # decrement this agent's cooldown at the start of the step
        if self.cooldown[agent_idx] > 0:
            self.cooldown[agent_idx] -= 1

        # 1. SPLAT enemy if adjacent and cooldown is 0
        if self.cooldown[agent_idx] == 0:
            for i, (rr, cc) in enumerate(positions):
                if i == agent_idx:
                    continue
                if int(teams[i]) != my_team and abs(int(rr) - r) + abs(int(cc) - c) == 1:
                    # use SPLAT and start cooldown
                    self.cooldown[agent_idx] = self.splat_cooldown
                    return SPLAT

        # 2. Move toward adjacent neutral or enemy tiles
        best_move = None
        best_score = -1
        for a, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W:
                tile = int(grid[nr, nc])
                # prefer neutral > enemy > own
                score = 2 if tile == -1 else (1 if tile != my_team else 0)
                if score > best_score:
                    best_score = score
                    best_move = a

        if best_move is not None and best_score > 0:
            return best_move

        # 3. Drift toward enemy side
        if my_team == 0 and c < self.W - 1:
            return RIGHT
        if my_team == 1 and c > 0:
            return LEFT

        # 4. Default: stay
        return STAY



def play_live(n_episodes=3, H=9, W=9, team_sizes=(2,2), max_steps=120):
    env = SplatoonGymEnv(H=H, W=W, team_sizes=team_sizes, max_steps=max_steps, render_mode="human")
    obs, info = env.reset()

    # Build models with correct obs_dim
    sample_vec = extract_local_obs(obs, 0, k=7)
    obs_dim = sample_vec.shape[0]
    cfg = IDQNConfig(obs_patch=7)

    team0 = TeamIDQN(obs_dim, cfg.n_actions, cfg)
    team1 = TeamIDQN(obs_dim, cfg.n_actions, cfg)

    team0.load("D:/University/COMP4900 multiagent reinforcement learning/checkpoints/team0_idqn.pt")
    team1.load("D:/University/COMP4900 multiagent reinforcement learning/checkpoints/team1_idqn.pt")

    teams = obs["teams"]
    n_agents = len(teams)
    heuristic = HeuristicController(H, W, n_agents, splat_cooldown=3)
    for ep in range(n_episodes):
        done = False
        while not done:
            acts = np.zeros(len(teams), dtype=np.int64)
            for i in range(len(teams)):
                vec = extract_local_obs(obs, i, k=cfg.obs_patch)
                if teams[i] == 0:
                    acts[i] = team0.act_greedy(vec)
                else:
                    acts[i] = heuristic.act_for_agent(obs, i)
                    #acts[i] = team1.act_greedy(vec)
                    #acts[i] = np.random.randint(0, 7)

            obs, r, term, trunc, info = env.step(acts)
            done = term or trunc
            # slow it slightly so spectators can watch
            time.sleep(0.02)
        obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    play_live()
