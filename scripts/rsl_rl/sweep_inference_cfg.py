# sweep config
sweep_config = {
    'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
    'metric': {'goal': 'minimize', 'name': 'Metrics/motion_data/error_imitation_metric'},
    'parameters': {
        'encoder_hidden_dims': {
            'values': [[128, 64, 32], [256, 128, 64], [512, 256, 128]]
        },
        'history_horizon': {
            'values': [15, 25, 35]
        },
        # 'resampling_time_range': {
        #     'values': [7.0, 3.0]
        # },
    } 
}

def update_config_from_sweep(env_cfg, agent_cfg, sweep_params):
        
    agent_cfg.policy.encoder_hidden_dims = sweep_params.encoder_hidden_dims             # list: [l1, l2, ..., lf]
    # agent_cfg.policy.latent_channels = sweep_params.latent_channels

    env_cfg.commands.motion_data.history_horizon = sweep_params.history_horizon         # int
    env_cfg.commands.motion_data.motion_dir = "motion_data_test" 
    # env_cfg.commands.motion_data.resampling_time_range = (sweep_params.resampling_time_range, sweep_params.resampling_time_range)       # tuple: (min, max)
    
    # agent_cfg.max_iterations = 10
    # agent_cfg.save_interval = 10
    return env_cfg, agent_cfg