import torch
from envs import VFEnv
import os
import utils
import shutil

from config import Configuration
from memoryreplay import MemoryReplay
from models import RLAgent
from trainer import Trainer


if __name__ == '__main__':

    # Get configuration
    config = Configuration("config.json")
    for seed in [100, 2023, 20000, 50000, 100000]:
        utils.fix_seed(seed)
        print('seed', seed)
        suffix = config.suffix
        path = './runs/{}/{}/'.format(config.type+"_"+suffix+"_zest", config.data_name)
        pred_path = './runs/'
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy('config.json', path)
        env = VFEnv(init_fos_std=config.init_fos_std, config=config)

        # Global initialization
        torch.cuda.init()
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Information about environments
        state_dim = env.state_dim
        action_dim = env.action_dim

        
        # Prepare Experience Memory Replay
        memory = MemoryReplay(config.capacity)

        model = RLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            target_update_freq=config.target_update_freq,
            learning_rate=config.lr,
            gamma=config.gamma,
            hidden_dim=config.hidden_dim,
            td_target=config.td_target,
            device=device
        )

        # Prepare Trainer
        trainer = Trainer(
            model=model,
            env=env,
            memory=memory,
            epsilon_start=config.epsilon_start,
            epsilon_final=config.epsilon_final,
            epsilon_decay=config.epsilon_decay,
            start_learning=config.start_learning,
            batch_size=config.batch_size,
            save_update_freq=config.save_update_freq,
            output_dir=path,
            config=config,
            phase='train',
            suffix=config.suffix + '_' + str(seed) + "_std" + str(config.stop_std)
        )

        if "mse" in config.suffix:
            trainer.loop('mse') # mse as reward
        elif "steps" in config.suffix:
            trainer.loop('steps') # num step as reward
        else:
            trainer.loop('shaping') # num step + mse as reward shaping
