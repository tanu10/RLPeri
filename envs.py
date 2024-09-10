import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib as mpl
mpl.use('Agg')
import os
import scipy.stats
from scipy.signal import *
from scipy.stats import norm


class VFEnv():
    def __init__(self, num_location=54, num_stimuli=41, init_fos_std=1, config=None)\
            -> None:
        '''Action is expected to be in the form [location,stimuli].
            stimuli value ranges from 0 to 40. 
            2D representation of state is 8x9
        '''
        self._NUM_LOC = num_location
        self._NUM_STIMULI = num_stimuli
        self.terminal = False
        self.name = "VFEnv"
        self._map_state = np.array([[-2, -2, -2, 0, 0, 0, 0, -2, -2],
                                     [-2, -2, 0, 0, 0, 0, 0, 0, -2],
                                     [-2, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [-2, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [-2, -2, 0, 0, 0, 0, 0, 0, -2],
                                     [-2, -2, -2, 0, 0, 0, 0, -2, -2]])
        self._map_state.flags.writeable = False
        self._init_state = np.array([[[0 for _ in range(9)] for _ in range(8)] for _ in range(num_stimuli)])
        self._init_state.flags.writeable = False

        self._map_action_location = np.argwhere(self._map_state != -2)
        self.state_mask = self._get_state_mask()
        self.state_mask.flags.writeable = False
        self.state_dim = np.shape(self._init_state)
        self.action_dim = np.array([self._NUM_LOC, self._NUM_STIMULI])
        '''std for FOS which has been initialized based on ground truth values'''
        self.init_fos_std = init_fos_std
        '''std of likelihood function for zest'''
        self.likelihood_std = config.likelihood_std
        '''std for stopping criterio of zest'''
        self.stop_std = config.stop_std

    def reset(self):
        return self._init_state.copy(), self._init_state.copy()

    def _get_state_mask(self):
        mask = np.array([[0 for _ in range(9)] for _ in range(8)])
        invalid_location = np.argwhere(self._map_state == -2)
        for _, [x,y] in enumerate(invalid_location):
            mask[x,y] = 1
        return mask
    
    def get_state_mask(self):
        return self.state_mask


    '''uses zest to complete testing at the given location. it keeps updating state_seen and state_not_seen.'''
    def step(self, loc, stimuli, gt, vals, state_seen, state_not_seen, init_pdfs):
        [x, y] = self._map_action_location[loc]
        init_fos = self.likelihood_func(gt[x,y], vals, respond=0, std=self.init_fos_std)
        min_threshold, max_threshold = 0, 40
        likelihood_std = self.likelihood_std
        terminate = False
        num_guess = 0
        new_pdf = init_pdfs[loc]
        guess = stimuli
        last_guess = guess
        while guess <= max_threshold and guess >= min_threshold and not terminate:  
            rand_prob = np.random.uniform(0, 1)
            respond = init_fos[guess] > rand_prob  # >0: yes, <0: no
            if respond:
                state_seen[guess, x, y] += 1
            else:
                state_not_seen[guess, x, y] += 1
            likelihood = self.likelihood_func(guess, vals, respond=respond, std=likelihood_std)
            new_pdf = np.multiply(new_pdf, likelihood)
            new_pdf = new_pdf / new_pdf.sum()
            new_std = np.sqrt((new_pdf * (vals - guess) ** 2).sum())
            terminate = new_std <= self.stop_std
            guess = np.argmax(new_pdf)
            num_guess += 1
            if guess >= max_threshold:
                guess = max_threshold
            if guess <= min_threshold:
                guess = min_threshold
            last_guess = guess
        return num_guess, last_guess, state_seen, state_not_seen, x, y

    def likelihood_func(self, threshold, vals, respond, std=1):
        likelihood = norm.cdf(vals, threshold, std)
        likelihood = respond*(respond-likelihood)+(1-respond)*likelihood
        return likelihood

    def get_pred_gt_mse(self, pred, gt, mask):
        gt = np.ma.masked_array(gt, mask)
        pred = np.ma.masked_array(pred, mask)
        mse = np.mean(np.square(gt - pred))
        return mse

    def get_gt_2d(self, gts):
        gts_2d = []
        for gt in gts:
            gt_2d = self._map_state.copy()
            for i, val in enumerate(gt):
                [x,y] = self._map_action_location[i]
                gt_2d[x,y] = val
            gts_2d.append(gt_2d)
        return gts_2d

def exponential_dist(presentation):
    mean_prob = 0.5
    lamda = np.log(mean_prob+1)/presentation
    print('lambda', lamda)
    p1= 0.25
    p2= 0.75
    q1 = np.log(p1 + 1) / lamda
    q2 = np.log(p2 + 1) / lamda
    iqr = q2-q1
    return iqr, q1, q2, lamda

def normal_dist(presentation):
    q1 = scipy.stats.norm.ppf(0.25)
    q2 = scipy.stats.norm.ppf(0.75)
    iqr = presentation*(q2-q1)
    std=0
    return iqr, q1, q2, std

def exp_dist_cdf(x, lamda):
    F=np.exp(x*lamda)-1
    return F

def norm_dist_cdf(x, mean, std):
    F=scipy.stats.norm.pdf(x,loc=mean, scale=std)
    return F

def plot_curve(pdf, vals, path, loc=0, mean=0, idx=None, iter=None, pres=None, is_cdf=False, title=''):
    plt.plot(vals,pdf, c='#bd0e3a', label='Probs')
    if is_cdf:
        X = [mean]*10
        Y= np.linspace(0, 0.5, 10)
        linestyle= 'dashed'
        plt.plot(X, Y, linestyle=linestyle, linewidth=1.5, color='black')
        X = np.linspace(vals[0], mean, mean/2)
        Y= [0.5]*int(mean//2)
        plt.plot(X, Y, linestyle=linestyle, linewidth=1.5, color='black')
        
    plt.xlim(0, 40)
    plt.ylim(-0.01, 1.01)
    plt.xlabel('Threshold')
    plt.ylabel('Probability')
    if idx:
        plt.title('{} curve loc {}'.format(title, loc))
    else:
        plt.title('{} curve sample {} loc {}'.format(title, idx, loc))
    if iter:
        plt.savefig(os.path.join(path, '{}_{}_{}_{}_{}.png'.format(title, idx, loc, iter, pres)))
    else:
        plt.savefig(os.path.join(path, '{}_{}.png'.format(title, loc)))
    plt.close()
