# sweep config
sweep_config = {
    'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
    'metric': {'goal': 'maximize', 'name': 'Train/mean_reward'},
    'parameters': {
        'encoder_hidden_dims': {
            'values': [[256, 128, 64], [256, 128, 64, 32], [512, 256, 128], [512, 256, 128, 64], [512, 256, 128, 64, 32]]
        },
        'history_horizon': {
            'values': [15, 20, 25]
        },
        # 'resampling_time_range': {
        #     'values': [7.0, 3.0]
        # },
    } 
}

def update_config_from_sweep(env_cfg, agent_cfg, sweep_params):
        
    agent_cfg.policy.encoder_hidden_dims = sweep_params.encoder_hidden_dims             # list: [l1, l2, ..., lf]
    
    env_cfg.commands.motion_data.history_horizon = sweep_params.history_horizon         # int
    # env_cfg.commands.motion_data.resampling_time_range = (sweep_params.resampling_time_range, sweep_params.resampling_time_range)       # tuple: (min, max)
    
    agent_cfg.max_iterations = 6000
    agent_cfg.save_interval = 2000
    return env_cfg, agent_cfg