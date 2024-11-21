# sweep config
sweep_config = {
    'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
    'parameters': {
        'policy_learning_rate': {
            'values': [5e-5, 1e-4, 1e-3]
        },
        'num_steps': {
            'values': [24, 48]
        },
        'history_horizon': {
            'values': [32, 64]
        },
        'forecast_horizon': {
            'values': [8, 16]
        },
        'entropy_coef': {
            'values': [0.0001, 0.0005, 0.001]
        },
    }
}

def update_config_from_sweep(env_cfg, agent_cfg, sweep_params):
    
    agent_cfg.algorithm.policy_learning_rate = sweep_params.policy_learning_rate
    agent_cfg.imagination.num_steps = sweep_params.num_steps
    
    env_cfg.observations.system_state_history.base_lin_vel.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_state_history.base_ang_vel.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_state_history.projected_gravity.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_state_history.joint_pos.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_state_history.joint_vel.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_state_history.joint_torque.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_action_history.pred_actions.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_extension_history.feet_current_contact_time.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_extension_history.feet_last_air_time.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_contact_history.body_contact.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon
    env_cfg.observations.system_termination_history.termination_flag.history_length = sweep_params.history_horizon + sweep_params.forecast_horizon

    env_cfg.observations.imagination_state_history.base_lin_vel.history_length = sweep_params.history_horizon
    env_cfg.observations.imagination_state_history.base_ang_vel.history_length = sweep_params.history_horizon
    env_cfg.observations.imagination_state_history.projected_gravity.history_length = sweep_params.history_horizon
    env_cfg.observations.imagination_state_history.joint_pos.history_length = sweep_params.history_horizon
    env_cfg.observations.imagination_state_history.joint_vel.history_length = sweep_params.history_horizon
    env_cfg.observations.imagination_state_history.joint_torque.history_length = sweep_params.history_horizon
    env_cfg.observations.imagination_action_history.pred_actions.history_length = sweep_params.history_horizon
    agent_cfg.system_dynamics.history_horizon = sweep_params.history_horizon
    
    agent_cfg.algorithm.entropy_coef = sweep_params.entropy_coef
    
    agent_cfg.max_iterations = 5000
    agent_cfg.save_interval = 500
    agent_cfg.imagination.num_envs = 4095
    return env_cfg, agent_cfg