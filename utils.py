import os
import numpy as np
import pandas as pd
import torch
import random
import h5py

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import animation

from scipy.ndimage.filters import gaussian_filter1d

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--random', action='store_true', default=False, help="test zest with random location and random stimuli")
    return parser.parse_args()

def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed

def _save(model, rewards, env_name, path, model_type, suffix):
    torch.save(model.state_dict(), os.path.join(path, 'model_{}{}.pt'.format(suffix, model_type)))
    plt.cla()
    plt.plot(rewards, c = '#bd0e3a', alpha = 0.3)
    plt.plot(gaussian_filter1d(rewards, sigma = 5), c = '#bd0e3a', label = 'Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.savefig(os.path.join(path, 'reward.png'))
    plt.close()

    pd.DataFrame(rewards, columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)

def save_checkpoint(agent, rewards, env_name, output_dir, suffix):
    _save(agent, rewards, env_name, output_dir, "_last", suffix)

def save_best( agent, rewards, env_name, output_dir, suffix):
    _save(agent, rewards, env_name, output_dir, "_best", suffix)

def load_model(model, path, seed, suffix, model_type):
    lower_model_path = os.path.join(path, 'model_{}_{}_{}.pt'.format(suffix, seed, model_type))
    print('load model ', lower_model_path)
    model.load_state_dict(torch.load(lower_model_path))
    return

def read_phase_dataset(data_path, phase, data_name, data_version):
    if data_name!="private":
        h5py_file = '{}/{}/v{}/{}_{}_v{}.h5py'.format(data_path, data_name, data_version, phase, data_name, data_version)
    else:
        h5py_file = '{}/{}/v{}/{}_v{}.h5py'.format(data_path, data_name, data_version, phase, data_version)
    data = h5py.File(h5py_file, 'r')
    vfs = data['labels']
    vfs = np.array(vfs)
    vfs[vfs<0]= 0
    vfs[vfs>40]=40
    return vfs

def read_vf_dataset(data_path, data_name, data_version):
    train_vfs = read_phase_dataset(data_path, 'train', data_name, data_version)
    test_vfs = read_phase_dataset(data_path, 'test', data_name, data_version)
    val_vfs = read_phase_dataset(data_path, 'val', data_name, data_version)
    return train_vfs, test_vfs, val_vfs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

