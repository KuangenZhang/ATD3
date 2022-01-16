import numpy as np
import os
import datetime
import cv2
import torch
import glob
import shutil
from utils import utils
from tqdm import tqdm
from scipy.stats import multivariate_normal
from tensorboardX import SummaryWriter
from methods import ATD3_RNN

class Solver(object):
    def __init__(self, args, env):
        print(args)
        self.args = args
        self.env = env
        self.file_name = ''
        self.result_path = "results"

        self.evaluations = []
        self.estimate_Q_vals = []
        self.Q1_vec = []
        self.Q2_vec = []
        self.true_Q_vals = []
        self.Q_ae_mean_vec = []
        self.Q_ae_std_vec = []


        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        # Initialize policy
        policy = ATD3_RNN.ATD3_RNN(state_dim, action_dim, max_action)
        self.policy = policy
        print('-------Current policy: {} --------------'.format(self.policy.__class__.__name__))
        self.replay_buffer = utils.ReplayBufferMat(max_size=args.max_timesteps)
        self.total_timesteps = 0
        self.pre_num_steps = self.total_timesteps
        self.timesteps_since_eval = 0
        self.timesteps_calc_Q_vale = 0
        self.best_reward = 0.0

        self.env_timeStep = 4

    def train_once(self):
        if self.total_timesteps != 0:
            self.policy.train(self.replay_buffer, self.args.batch_size, self.args.discount,
                              self.args.tau, self.args.policy_noise, self.args.noise_clip,
                              self.args.policy_freq)

    def eval_once(self):
        self.pbar.update(self.total_timesteps - self.pre_num_steps)
        self.pre_num_steps = self.total_timesteps

        # Evaluate episode
        if self.timesteps_since_eval >= self.args.eval_freq:
            self.timesteps_since_eval %= self.args.eval_freq
            avg_reward = evaluate_policy(self.env, self.policy, self.args)
            self.evaluations.append(avg_reward)
            self.writer_test.add_scalar('ave_reward', avg_reward, self.total_timesteps)

            if self.args.save_all_policy:
                self.policy.save(
                    self.file_name + str(int(int(self.total_timesteps/self.args.eval_freq)* self.args.eval_freq)),
                    directory=self.log_dir)

            np.save(self.log_dir + "/test_accuracy", self.evaluations)
            utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
            if self.best_reward < avg_reward:
                self.best_reward = avg_reward
                print("Best reward! Total T: %d Episode T: %d Reward: %f" %
                      (self.total_timesteps, self.episode_timesteps, avg_reward))
                self.policy.save(self.file_name, directory=self.log_dir)

    def reset(self):
        # Reset environment
        self.obs = self.env.reset()
        self.obs_vec = np.dot(np.ones((self.args.seq_len, 1)), self.obs.reshape((1, -1)))
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.still_steps = 0

    def train(self):
        # Evaluate untrained policy
        self.evaluations = [evaluate_policy(self.env, self.policy, self.args)]
        self.log_dir = '{}/{}'.format(self.result_path, self.args.log_path)
        print("---------------------------------------")
        print("Settings: %s" % self.log_dir)
        print("---------------------------------------")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # TesnorboardX
        self.writer_test = SummaryWriter(logdir=self.log_dir)
        self.pbar = tqdm(total=self.args.max_timesteps, initial=self.total_timesteps, position=0, leave=True)
        done = True
        while self.total_timesteps < self.args.max_timesteps:
            # ================ Train =============================================#
            self.train_once()
            # ====================================================================#
            if done:
                self.eval_once()
                self.reset()
                done = False
            # Select action randomly or according to policy
            if self.total_timesteps < self.args.start_timesteps:
                action = self.env.action_space.sample()
                p = 1
            else:
                if 'RNN' in self.args.policy_name:
                    action = self.policy.select_action(np.array(self.obs_vec))
                elif 'SAC' in self.args.policy_name or 'HRL' in self.args.policy_name:
                    action = self.policy.select_action(np.array(self.obs), eval=False)
                else:
                    action = self.policy.select_action(np.array(self.obs))

                noise = np.random.normal(0, self.args.expl_noise,
                                         size=self.env.action_space.shape[0])
                if self.args.expl_noise != 0:
                    action = (action + noise).clip(
                        self.env.action_space.low, self.env.action_space.high)

                if 'HRL' in self.args.policy_name:
                    p_noise = multivariate_normal.pdf(
                        noise, np.zeros(shape=self.env.action_space.shape[0]),
                        self.args.expl_noise * self.args.expl_noise * np.identity(noise.shape[0]))
                    if 'SHRL' in self.args.policy_name:
                        p = (p_noise * utils.softmax(self.policy.option_prob))[0]
                    else:
                        p = (p_noise * utils.softmax(self.policy.q_predict)[self.policy.option_val])[0]

            state_id = 0
            # Perform action
            new_obs, reward, done, _ = self.env.step(action)

            self.episode_reward += reward

            done_bool = 0 if self.episode_timesteps + 1 == self.env._max_episode_steps else float(done)

            if 'RNN' in self.args.policy_name:
                # Store data in replay buffer
                new_obs_vec = utils.fifo_data(np.copy(self.obs_vec), new_obs)
                self.replay_buffer.add((np.copy(self.obs_vec), new_obs_vec, action, reward, done_bool, state_id))
                self.obs_vec = utils.fifo_data(self.obs_vec, new_obs)
            else:
                self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool, state_id))

            self.obs = new_obs
            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1
            self.timesteps_calc_Q_vale += 1

        # Final evaluation
        self.eval_once()
        self.env.reset()

    def eval_only(self, is_reset = True):
        video_dir = '{}/video_all'.format(self.result_path)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        model_path_vec = glob.glob(self.result_path + '/{}'.format(self.args.log_path))
        print(model_path_vec)
        for model_path in model_path_vec:
            self.policy.load("%s" % (self.file_name + self.args.load_policy_idx), directory=model_path)
            for _ in range(1):
                if self.args.save_video:
                    video_name = video_dir + '/{}_{}_{}.mp4'.format(
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        self.file_name, self.args.load_policy_idx)
                obs = self.env.reset()
                if 'RNN' in self.args.policy_name:
                    obs_vec = np.dot(np.ones((self.args.seq_len, 1)), obs.reshape((1, -1)))

                obs_mat = np.asarray(obs)
                done = False

                if self.args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    img = self.env.render(mode='rgb_array')
                    out_video = cv2.VideoWriter(video_name, fourcc, 50.0, (img.shape[1], img.shape[0]))

                while not done:
                    if 'RNN' in self.args.policy_name:
                        action = self.policy.select_action(np.array(obs_vec))
                    else:
                        action = self.policy.select_action(np.array(obs))

                    obs, reward, done, _ = self.env.step(action)

                    if 'RNN' in self.args.policy_name:
                        obs_vec = utils.fifo_data(obs_vec, obs)

                    if 0 != self.args.state_noise:
                        obs[8:20] += np.random.normal(0, self.args.state_noise, size=obs[8:20].shape[0]).clip(
                            -1, 1)

                    obs_mat = np.c_[obs_mat, np.asarray(obs)]

                    if self.args.save_video:
                        img = self.env.render(mode='rgb_array')
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out_video.write(img)
                    elif self.args.render:
                        self.env.render(mode='rgb_array')

                if self.args.save_video:
                    out_video.release()
        if is_reset:
            self.env.reset()

# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, args, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        if 'RNN' in args.policy_name:
            obs_vec = np.dot(np.ones((args.seq_len, 1)), obs.reshape((1, -1)))
        done = False
        while not done:
            if 'RNN' in args.policy_name:
                action = policy.select_action(np.array(obs_vec))
            else:
                action = policy.select_action(np.array(obs))

            obs, reward, done, _ = env.step(action)
            if 'RNN' in args.policy_name:
                obs_vec = utils.fifo_data(obs_vec, obs)
            avg_reward += reward
    avg_reward /= eval_episodes
    # print ("---------------------------------------"                      )
    # print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    # print ("---------------------------------------"                      )
    return avg_reward
