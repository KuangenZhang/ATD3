import os
import sys
import pybullet_envs, gym
import argparse
import numpy as np
from utils.solver import utils, Solver

def test_env(env):
    env.reset()
    state = np.random.rand(22)
    print(env.set_robot(state) - state)
    while True:
        env.render()

def convert_args_to_bool(args):
    args.eval_only = (args.eval_only in ['True', True])
    args.render = (args.render in ['True', True])
    args.save_video = (args.save_video in ['True', True])
    args.save_all_policy = (args.save_all_policy in ['True', True])
    return args

def main(args):
    env = gym.make(args.env_name)
    if args.render:
        env.render('human')
    solver = Solver(args, env)
    if not args.eval_only:
        solver.train()
    else:
        solver.eval_only()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default='ATD3_RNN')  # Policy name
    parser.add_argument("--env_name", default="Walker2DBulletEnv-v0")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs')

    parser.add_argument("--eval_only", default=True)
    parser.add_argument("--render", default=True)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--save_all_policy", default=False)
    parser.add_argument("--load_policy_idx", default='')
    parser.add_argument("--reward_name", default='')
    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--option_num", default=4, type=int)
    parser.add_argument("--buffer_size", default=3e5, type=int)
    parser.add_argument("--lr", default=1e-3, type=int)
    parser.add_argument("--seed", default=1, type=int) # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate.
    # 5e3 for not evaluating Q value; 3e4 for evaluating Q value
    parser.add_argument("--max_timesteps", default=3e5, type=int)  # Max time steps to run environment for

    parser.add_argument("--disc_ratio", default=0.2, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=0, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates

    args = parser.parse_args()
    args = convert_args_to_bool(args)
    main(args)

