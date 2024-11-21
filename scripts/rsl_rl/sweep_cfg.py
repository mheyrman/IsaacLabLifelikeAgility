# sweep config
sweep_config = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'metric': {'goal': 'maximize', 'name': 'Train/mean_reward'},
    'parameters': {
        'learning_rate': {
            'values': [5e-5, 1e-4, 1e-3]
        },
        'entropy_coef': {
            'values': [0.0001, 0.0005, 0.001]
        },
        'encoder_hidden_dims': {
            'values': [[256, 128, 64], [256, 128, 128, 64], [256, 128, 128, 64, 64], [256, 256, 128, 128, 64, 64], [128, 128, 64, 64], [64, 64, 64, 64, 64, 64], [32, 32, 32, 32, 32, 32]]
        },
        'history_horizon': {
            'values': [1, 5, 10, 15, 25]
        },
        'resampling_time_range': {
            'values': [(5.0, 5.0), (10.0, 10.0)]
        },
    } 
}

def update_config_from_sweep(env_cfg, agent_cfg, sweep_params):
    
    # agent_cfg.algorithm.learning_rate = sweep_params.learning_rate
    # agent_cfg.algorithm.entropy_coef = sweep_params.entropy_coef
    
    agent_cfg.policy.encoder_hidden_dims = sweep_params.encoder_hidden_dims             # list: [l1, l2, ..., lf]
    
    env_cfg.commands.motion_data.history_horizon = sweep_params.history_horizon         # int
    env_cfg.commands.motion_data.resampling_time_range = sweep_params.resampling_time_range   # tuple: (min, max)
    
    agent_cfg.max_iterations = 10
    agent_cfg.save_interval = 10
    return env_cfg, agent_cfg