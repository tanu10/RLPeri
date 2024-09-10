import numpy as np
from torch.utils import tensorboard
import os
from utils import save_checkpoint, save_best, AverageMeter, read_vf_dataset
from tools import visualize
from envs import plot_curve
from torch import nn
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random


class Trainer:
    def __init__(self, model, env, memory, epsilon_start, epsilon_final,
                 epsilon_decay, start_learning, batch_size, save_update_freq, output_dir, 
                 config, phase, output_path=None, suffix=None):
        self.model = model
        self.env = env
        self.memory = memory
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.start_learning = start_learning
        self.batch_size = batch_size
        self.save_update_freq = save_update_freq
        self.gamma = config.gamma
        self.output_dir = output_dir
        self.epsilon = epsilon_start
        data_path = config.data_path
        self.data_name = config.data_name
        data_version = config.data_version
        self.type = config.type
        self.train_gt, self.test_gt, self.val_gt, train_gt = self._get_gts(data_path, data_version)
        self.vals = np.arange(0, 41)
        self.init_pdfs = self._initialize_pdf(train_gt)        
        self.stop_std = config.stop_std
        self.output_path = output_path
        self.suffix = suffix
        print('suffix', suffix)
        print('output_dir', output_dir)
        self.phase = phase

    def _initialize_pdf(self, dataset):
        dataset = np.array(dataset)
        vals = np.arange(-1, 41)
        init_pdf = []
        path = './output/test/init_data_pdf/'
        if not os.path.exists(path):
            os.makedirs(path)
        for i, loc_data in enumerate(dataset.T):
            init_loc = np.histogram(loc_data, vals, (-1, 41), normed=True)
            init_loc = init_loc[0]
            # plot_curve(init_loc, self.vals, path, i, title='init_dataset_pdf')
            init_pdf.append(init_loc)
        return init_pdf

    def _get_gts(self, data_path, data_version):
        train_gt, test_gt, val_gt = read_vf_dataset(data_path, self.data_name, data_version) 
        train_gt_2d = self.env.get_gt_2d(train_gt)
        val_gt = self.env.get_gt_2d(val_gt)
        test_gt = self.env.get_gt_2d(test_gt)
        return train_gt_2d, val_gt, test_gt, train_gt

    def _exploration(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_final:
            self.epsilon = self.epsilon_final
        return self.epsilon

    def loop(self, reward_type):
        state_seen, state_not_seen = self.env.reset()
        episode_reward = 0
        episode_idx = 0
        last_best_rw = 1e3
        all_rewards = []
        log_dir = '{}/logs_learn/'.format(self.output_dir)
        w = tensorboard.SummaryWriter(log_dir)
        print('Start training')
        no_loop = 2
        total_iter = len(self.train_gt)*no_loop
        pbar = tqdm.tqdm(range(total_iter))
        pbar.set_description("Training")
        step = 0
        for _ in range(no_loop):
            np.random.shuffle(self.train_gt)
            for gt in self.train_gt:
                pred = np.array([[-2, -2, -2, -1, -1, -1, -1, -2, -2],
                          [-2, -2, -1, -1, -1, -1, -1, -1, -2],
                          [-2, -1, -1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                          [-2, -1, -1, -1, -1, -1, -1, -1, -1],
                          [-2, -2, -1, -1, -1, -1, -1, -1, -2],
                          [-2, -2, -2, -1, -1, -1, -1, -2, -2]])
                epsilon = self._exploration()
                done = False
                act_batch = []
                reward_batch = []
                potential_batch = [0]
                state_seen_batch = []
                state_not_seen_batch = []
                next_state_seen_batch = []
                next_state_not_seen_batch = []
                term_batch = []
                gt_batch = []
                episode_reward = 0
                action_mask = [0 for _ in range(self.env.action_dim[0])]
                
                n_step = 0
                done = False
                while not done:
                    step += 1
                    if step % self.start_learning == 0:
                        self.model.train()
                        st, st_n, loc, r, nst, nst_n, term = self.memory.sample(self.batch_size)
                        lloss = self.model.update_policy(st, st_n, loc, r, nst, nst_n, term)
                        w.add_scalar("loss/lloss", lloss, global_step=step)
        
                    state_seen_batch.append(state_seen.copy())
                    state_not_seen_batch.append(state_not_seen.copy())
                    gt_batch.append(gt.copy())
                    if np.random.random() > epsilon:
                        loc, stimuli = self.model.get_action([state_seen/10], [state_not_seen/10], action_mask)
                    else:
                        loc_list = []
                        for i, mask in enumerate(action_mask):
                            if mask == 0:
                                loc_list.append(i)
                        loc = np.random.choice(loc_list)
                        stimuli = np.random.choice([i for i in range(self.env._NUM_STIMULI)])

                    assert action_mask[loc] == 0, "Discovering an already discovered location !!!"
                    
                    zest_step, guess, zest_state_seen, zest_state_not_seen, x, y = self.env.step(loc, stimuli, gt.copy(), self.vals, 
                            state_seen.copy(), state_not_seen.copy(), self.init_pdfs)
                    episode_reward += zest_step
                    n_step += zest_step
                    pred[x, y] = guess
                    action_mask[loc] = 1 # location has been tested
                    state_seen = zest_state_seen
                    state_not_seen = zest_state_not_seen
                    next_state_seen_batch.append(zest_state_seen.copy())
                    next_state_not_seen_batch.append(zest_state_not_seen.copy())
                    act_batch.append([loc, stimuli])
                    mse = self.env.get_pred_gt_mse(pred.copy(), gt.copy(), self.env.get_state_mask())
                    if reward_type == "mse":
                        reward_batch.append((350-mse)/300) # to convert reward into a positive value
                    else:
                        reward_batch.append((10-zest_step)/10) # to convert reward into a positive value
                    potential_batch.append((250-mse)/200)
                    if sum(action_mask) == self.env.action_dim[0]: # if all the locations have been tested
                        done = True
                    term_batch.append(0 if done else 1)
                
                if reward_type == "shaping":
                    for i in range(1,len(reward_batch)):
                        reward_batch[i] = reward_batch[i] + self.gamma * potential_batch[i+1] - potential_batch[i]
                

                for st, st_n, act, r, nst, nst_n, term in zip(state_seen_batch, state_not_seen_batch, act_batch,
                                                                reward_batch, next_state_seen_batch, next_state_not_seen_batch, term_batch):

                    st, st_n, nst, nst_n = st/10, st_n/10, nst/10, nst_n/10
                    self.memory.push((st, st_n, [act], [r/10], nst, nst_n, [term]))

                episode_reward = n_step
                state_seen, state_not_seen = self.env.reset()
                all_rewards.append(episode_reward)
                ms = "Reward on Episode [{}/{}]: {} {:.2f} {:.3f} {:.2f}".format(
                    episode_idx, total_iter, n_step, mse, epsilon, (350-mse)/300)
                pbar.set_description(ms)
                pbar.update(1)
                w.add_scalar("reward/episode_reward",episode_reward, global_step=len(all_rewards))

                episode_idx += 1
                if episode_idx % self.save_update_freq == 0:
                    avg_mse, avg_steps = self.test("validate")
                    if reward_type=="mse" and last_best_rw > avg_mse:
                        last_best_rw = avg_mse
                        save_best(self.model, all_rewards,self.env.name, self.output_dir, self.suffix)
                    if reward_type == "steps" and last_best_rw > avg_steps:
                        last_best_rw = avg_steps
                        save_best(self.model, all_rewards,self.env.name, self.output_dir, self.suffix)
                    if reward_type=="shaping" and last_best_rw > 0.01*avg_steps+avg_mse:
                        last_best_rw = 0.01*avg_steps+avg_mse
                        save_best(self.model, all_rewards,self.env.name, self.output_dir, self.suffix)
            save_checkpoint(self.model,all_rewards, self.env.name, self.output_dir, self.suffix)
        w.close()

    def test(self, test_type="test", rand=None):
        all_rewards = []
        all_actions = []
        all_2d_rewards = []
        inits = []
        all_final_guesses = []
        mses = AverageMeter()
        mse_list = []
        self.model.eval()
        vis = True
        if test_type == "validate":
            pbar = tqdm.tqdm(range(len(self.val_gt)))
            pbar.set_description("Validating")
            gts = self.val_gt
            vis = False
        else:
            pbar = tqdm.tqdm(range(len(self.test_gt)))
            pbar.set_description("Testing")
            gts = self.test_gt

        for gt in gts:
            state_seen, state_not_seen = self.env.reset()
            pred = np.array([[-2, -2, -2, -1, -1, -1, -1, -2, -2],
                                  [-2, -2, -1, -1, -1, -1, -1, -1, -2],
                                  [-2, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-2, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-2, -2, -1, -1, -1, -1, -1, -1, -2],
                                  [-2, -2, -2, -1, -1, -1, -1, -2, -2]])
            rew_2d = pred.copy()
            init_2d = pred.copy()
            episode_reward = 0
            done = False
            actions = []
            rewards = []
            final_guess = []
            action_mask = [0 for _ in range(self.env.action_dim[0])]
            step = 0
            while not done:
                loc, stimuli = self.model.get_action([state_seen/10], [state_not_seen/10], action_mask)
                '''for random action'''
                if rand:
                    loc_list = []
                    for i, mask in enumerate(action_mask):
                        if mask == 0:
                            loc_list.append(i)
                    loc = np.random.choice(loc_list)
                    stimuli = np.random.choice([i for i in range(self.env.action_dim[1])])
                zest_step, guess, state_seen, state_not_seen, x, y = self.env.step(loc, stimuli, gt.copy(), self.vals,
                                                state_seen.copy(), state_not_seen.copy(), self.init_pdfs)
                episode_reward += zest_step
                step += zest_step
                # final estimated value
                pred[x, y] = guess
                rew_2d[x, y] = zest_step
                init_2d[x, y] = stimuli
                assert action_mask[loc] == 0, "Discovering an already discovered location !!!"

                actions.append([loc, stimuli])
                action_mask[loc] = 1
                final_guess.append(guess)

                if sum(action_mask) == self.env.action_dim[0]:
                    action_mask = [0 for _ in range(self.env.action_dim[0])]
                    done = True
                if done:
                    mse = self.env.get_pred_gt_mse(pred.copy(), gt.copy(), self.env.get_state_mask())
                    state_seen, state_not_seen = self.env.reset()
                    all_rewards.append(step)
                    ms = "Reward on Episode {}: {} {:.2f} {:.2f} {:.2f}".format(
                        len(all_rewards), step, mse, np.mean(all_rewards), mses.avg)

                    pbar.set_postfix_str(ms)
                    pbar.update(1)
            mses.update(mse, 1)
            mse_list.append(mse)
            inits.append(init_2d)
            all_actions.append(actions)
            all_2d_rewards.append(rew_2d)
            all_final_guesses.append(pred)
        avg_rw = np.mean(all_rewards)
        print('avg reward', avg_rw, 'avg MSE', mses.avg)
        if vis:
            vis_step = False  # if True, visualize step by step taking action
            visualize(all_actions, all_2d_rewards, all_rewards, all_final_guesses,
                      gts, inits, self.output_path, self.suffix, mse_list, vis_step)
        return mses.avg, avg_rw
