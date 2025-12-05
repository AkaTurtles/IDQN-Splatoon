# main_train.py
from splatoon import SplatoonGymEnv  #
from idqn import train_idqn

if __name__ == "__main__":
    env = SplatoonGymEnv(H=9, W=9, team_sizes=(2,2), max_steps=200, render_mode=None)
    team0, team1 = train_idqn(env, total_steps=80_000, device="cpu", obs_patch=7, log_every=2000, seed=0, plot_learning_curve=True)
    