import json

class Configuration:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Exploration
        self.epsilon_start = config['exploration']['epsilon_start']
        self.epsilon_final = config['exploration']['epsilon_final']
        self.epsilon_decay = config['exploration']['epsilon_decay']

        # Training
        self.target_update_freq = config['training']['target_update_freq']
        self.start_learning = config['training']['start_learning']
        self.lr = config['training']['learning_rate']

        # Memory replay
        self.capacity = config['memory_replay']['capacity']
        self.batch_size = config['memory_replay']['batch_size']

        # Output
        self.save_update_freq = config['output']['save_update_freq']

        # Model
        self.td_target = config['model']['temporal_difference_target']
        assert self.td_target in ("mean", "max", "individual")
        self.gamma = config['model']['gamma']
        self.hidden_dim = config['model']['hidden_dim']

        # Device
        self.device = config['device']

        # Params
        self.data_name = config['data_name']
        self.data_version = config['data_version']
        self.data_path = config['data_path']
        self.ground_truth = config['ground_truth']
        self.output_path = config['output_path']
        self.suffix = config['suffix']
        self.init_fos_std = config['init_fos_std']
        self.likelihood_std = config['likelihood_std']
        self.stop_std = config['stop_std']
        self.type = config['type']
