import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
import numpy as np

try:
    import pygame
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False

# Actions (per agent)
STAY, UP, DOWN, LEFT, RIGHT, SPRAY, SPLAT = range(7)
DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

@dataclass
class AgentState:
    r: int
    c: int
    team: int           # 0 or 1
    stunned: bool = False
    last_dir: tuple = (0, 0)

class SplatoonGymEnv(gym.Env):
    """
    Gymnasium environment for a minimal Splatoon-like MARL game
    with Pygame rendering.

    - Joint action step: MultiDiscrete([7]*n_agents)
    - Observation: Dict with grid & agent states
    - Reward: np.ndarray of per-agent rewards (vector)
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 8}

    def __init__(self, H=11, W=11, team_sizes=(2, 2), seed=None,
                 reward_neutral=1.0, reward_overwrite=0.5, penalty_splatted=-1.0,
                 use_team_shaping=False, shaping_alpha=0.0, max_steps=200,
                 render_mode=None, tile_px=36, grid_line=1):
        super().__init__()
        self.H, self.W = H, W
        self.team_sizes = team_sizes
        self.n_agents = team_sizes[0] + team_sizes[1]
        self.teams = np.array([0]*team_sizes[0] + [1]*team_sizes[1], dtype=np.int8)

        # Rewards
        self.r_neut = float(reward_neutral)
        self.r_over = float(reward_overwrite)
        self.p_splat = float(penalty_splatted)
        self.use_team_shaping = bool(use_team_shaping)
        self.alpha = float(shaping_alpha)

        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Spaces
        self.action_space = spaces.MultiDiscrete([7] * self.n_agents)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-1, high=1, shape=(H, W), dtype=np.int8),
            "positions": spaces.Box(low=0, high=max(H, W), shape=(self.n_agents, 2), dtype=np.int16),
            "stunned": spaces.MultiBinary(self.n_agents),
            "teams": spaces.MultiBinary(self.n_agents),
            "t": spaces.Discrete(self.max_steps + 1),
        })

        # State
        self.grid = None
        self.agents = None
        self.t = 0
        self._last_reward_vec = np.zeros(self.n_agents, dtype=np.float32)

        # --- Rendering config ---
        self._win = None
        self._clock = None
        self._surf = None
        self.tile_px = int(tile_px)
        self.grid_line = int(grid_line)
        # Colors
        self._COLORS = {
            "bg": (18, 18, 18),
            "grid": (35, 35, 40),
            "neutral": (110, 110, 120),
            "A": (51, 132, 255),       # blue
            "B": (255, 140, 0),        # orange
            "agent_edge": (20, 20, 20),
            "text": (230, 230, 235),
            "stun": (240, 50, 80),
        }

    # -------------- Gym API -------------- #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.grid = np.full((self.H, self.W), -1, dtype=np.int8)
        self.agents = []
        # Place teams on opposite sides
        for team in self.teams:
            r = self.np_random.integers(0, self.H)
            if team == 0:
                c = self.np_random.integers(0, self.W // 3)
            else:
                c = self.np_random.integers(2 * self.W // 3, self.W)
            self.agents.append(AgentState(r=r, c=c, team=int(team)))
            self.grid[r, c] = int(team)
        self.t = 0
        self._last_reward_vec[:] = 0
        obs = self._obs()
        info = {"coverage": self._team_coverage()}
        if self.render_mode == "human":
            self._ensure_pygame()
            self.render()
        return obs, info

    def step(self, action):
        """
        action: np.ndarray shape (n_agents,) with ints in {0..6}
        Returns: (obs, reward_vec, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.int64)
        assert action.shape == (self.n_agents,)

        prev_cov = self._team_coverage()

        # 1) Stun resolution (stunned -> forced STAY this step)
        eff_action = action.copy()
        for i, ag in enumerate(self.agents):
            if ag.stunned:
                eff_action[i] = STAY

        # 2) Propose moves
        proposals = {}  # cell -> list(agent_idx)
        new_pos = [(ag.r, ag.c) for ag in self.agents]
        for i, a in enumerate(eff_action):
            ag = self.agents[i]
            if a in DIRS:
                dr, dc = DIRS[a]
                nr, nc = ag.r + dr, ag.c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    new_pos[i] = (nr, nc)
                    proposals.setdefault((nr, nc), []).append(i)
                    ag.last_dir = (dr, dc)
                else:
                    new_pos[i] = (ag.r, ag.c)  # wall -> stay
            elif a == STAY:
                new_pos[i] = (ag.r, ag.c) # SPRAY/SPLAT keep position

        original = {i: (ag.r, ag.c) for i, ag in enumerate(self.agents)}
        intended = {i: new_pos[i] for i in range(self.n_agents)}

        # 3) Resolve collisions & swaps
        blocked = set()
        for cell, idxs in proposals.items():
            if len(idxs) > 1:
                blocked.update(idxs)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if intended[i] == original[j] and intended[j] == original[i]:
                    blocked.add(i); blocked.add(j)

        # 4) Apply moves
        for i, ag in enumerate(self.agents):
            if i not in blocked and eff_action[i] in DIRS or eff_action[i] == STAY:
                r, c = intended[i] if i not in blocked else original[i]
                ag.r, ag.c = r, c

        # 5) Painting from occupancy
        cell_owner_before = self.grid.copy()
        paint_credit = np.zeros(self.n_agents, dtype=np.float32)
        for i, ag in enumerate(self.agents):
            paint_credit[i] += self._paint_cell(ag.r, ag.c, ag.team, cell_owner_before)

        # 6) SPRAY painting (uses last_dir; if none, spray own cell)
        for i, a in enumerate(eff_action):
            if a == SPRAY:
                ag = self.agents[i]
                dr, dc = ag.last_dir
                rr, cc = (ag.r + dr, ag.c + dc) if (dr or dc) else (ag.r, ag.c)
                if 0 <= rr < self.H and 0 <= cc < self.W:
                    paint_credit[i] += self._paint_cell(rr, cc, ag.team, cell_owner_before)

        # 7) SPLAT (applies stun for next step)
        splatted = np.zeros(self.n_agents, dtype=bool)
        for i, a in enumerate(eff_action):
            if a == SPLAT:
                ai = self.agents[i]
                for j, aj in enumerate(self.agents):
                    if aj.team != ai.team and abs(aj.r - ai.r) + abs(aj.c - ai.c) == 1:
                        aj.stunned = True
                        splatted[j] = True

        # Clear older stun flags if not freshly splatted
        for j, aj in enumerate(self.agents):
            if not splatted[j]:
                aj.stunned = False

        # 8) Rewards (vector)
        reward = paint_credit + self.p_splat * splatted.astype(np.float32)

        if self.use_team_shaping:
            delta = self._team_coverage() - prev_cov
            for i, ag in enumerate(self.agents):
                reward[i] += self.alpha * delta[ag.team]

        self._last_reward_vec = reward.copy()

        # Termination / truncation
        self.t += 1
        terminated = False  # define if you want: e.g., no neutral tiles left
        truncated = self.t >= self.max_steps

        obs = self._obs()
        info = {"coverage": self._team_coverage()}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # -------------- Helpers -------------- #
    def _paint_cell(self, r, c, team, before):
        prev = self.grid[r, c]
        self.grid[r, c] = team
        if prev == -1:
            return self.r_neut
        elif prev != team:
            return self.r_over
        return 0.0

    def _team_coverage(self):
        total = self.H * self.W
        a = np.count_nonzero(self.grid == 0)
        b = np.count_nonzero(self.grid == 1)
        return np.array([a / total, b / total], dtype=np.float32)

    def _obs(self):
        positions = np.array([[ag.r, ag.c] for ag in self.agents], dtype=np.int16)
        stunned = np.array([ag.stunned for ag in self.agents], dtype=np.int8)
        obs = {
            "grid": self.grid.copy(),
            "positions": positions,
            "stunned": stunned,
            "teams": self.teams.copy(),
            "t": int(self.t),
        }
        return obs

    # -------------- Rendering -------------- #
    def _ensure_pygame(self):
        if not _HAS_PYGAME:
            raise RuntimeError("pygame is not installed. `pip install pygame`")
        if self._win is None:
            pygame.init()
            w = self.W * self.tile_px
            h = self.H * self.tile_px + 60  # HUD bar
            self._win = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Splatoon MARL (Gymnasium)")
            self._clock = pygame.time.Clock()
            self._surf = pygame.Surface((w, h))

    def render(self):
        if self.render_mode == "ansi":
            # simple text render to stdout
            chars = {-1: ".", 0: "A", 1: "B"}
            grid_str = "\n".join("".join(chars[int(x)] for x in row) for row in self.grid)
            print(grid_str)
            return

        if self.render_mode not in ("human", "rgb_array"):
            return

        self._ensure_pygame()
        self._handle_events()

        tpx = self.tile_px
        gl = self.grid_line
        Acol = self._COLORS["A"]; Bcol = self._COLORS["B"]
        Neu = self._COLORS["neutral"]; bg = self._COLORS["bg"]; gridc = self._COLORS["grid"]
        textc = self._COLORS["text"]; stun_c = self._COLORS["stun"]; edge = self._COLORS["agent_edge"]

        self._surf.fill(bg)

        # Draw tiles
        for r in range(self.H):
            for c in range(self.W):
                val = int(self.grid[r, c])
                color = Neu if val == -1 else (Acol if val == 0 else Bcol)
                pygame.draw.rect(self._surf, color, (c*tpx, r*tpx, tpx, tpx))
                if gl:
                    pygame.draw.rect(self._surf, gridc, (c*tpx, r*tpx, tpx, tpx), gl)

        # Draw agents
        font = pygame.font.SysFont(None, 18)
        for i, ag in enumerate(self.agents):
            cx = ag.c * tpx + tpx//2
            cy = ag.r * tpx + tpx//2
            fill = Acol if ag.team == 0 else Bcol
            pygame.draw.circle(self._surf, edge, (cx, cy), int(tpx*0.42))
            pygame.draw.circle(self._surf, fill, (cx, cy), int(tpx*0.38))
            # index label
            label = font.render(str(i), True, (10, 10, 10))
            rect = label.get_rect(center=(cx, cy))
            self._surf.blit(label, rect)
            # stunned "X"
            if ag.stunned:
                s = int(tpx*0.28)
                pygame.draw.line(self._surf, stun_c, (cx-s, cy-s), (cx+s, cy+s), 3)
                pygame.draw.line(self._surf, stun_c, (cx+s, cy-s), (cx-s, cy+s), 3)

        # HUD
        hud_y = self.H * tpx
        pygame.draw.rect(self._surf, (25,25,28), (0, hud_y, self.W*tpx, 60))
        big = pygame.font.SysFont(None, 22)
        cov = self._team_coverage()
        hud_text = f"Step: {self.t}   Coverage A: {cov[0]*100:.1f}%   B: {cov[1]*100:.1f}%"
        self._surf.blit(big.render(hud_text, True, textc), (8, hud_y+6))
        # Last rewards (per agent)
        rtxt = "Rewards: " + "  ".join([f"{i}:{self._last_reward_vec[i]:+.2f}" for i in range(self.n_agents)])
        self._surf.blit(font.render(rtxt, True, textc), (8, hud_y+32))

        if self.render_mode == "human":
            self._win.blit(self._surf, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return pygame.surfarray.array3d(self._surf).swapaxes(0, 1)

    def _handle_events(self):
        # allow window close without hanging the process
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if _HAS_PYGAME:
            try:
                if self._win is not None:
                    pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
        self._win = None
        self._clock = None
        self._surf = None

if __name__ == "__main__":
    env = SplatoonGymEnv(H=9, W=9, team_sizes=(2,2), max_steps=120, render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    while True:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            obs, info = env.reset()
