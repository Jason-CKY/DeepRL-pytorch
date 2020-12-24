from Algorithms.ddpg.core import MLPActorCritic, CNNActorCritic

def get_actor_critic_module(ac_kwargs, RL_Algorithm):
    if ac_kwargs['model_type'].lower() == 'mlp':
        if RL_Algorithm.lower() == 'ddpg':
            from Algorithms.ddpg.core import MLPActorCritic            
            return MLPActorCritic
        elif RL_Algorithm.lower() == 'td3':
            from Algorithms.td3.core import MLPActorCritic            
            return MLPActorCritic
        elif RL_Algorithm.lower() == 'trpo':
            from Algorithms.trpo.core import MLPActorCritic            
            return MLPActorCritic
        elif RL_Algorithm.lower() == 'ppo':
            from Algorithms.ppo.core import MLPActorCritic            
            return MLPActorCritic

    elif ac_kwargs['model_type'].lower() == 'cnn':
        if RL_Algorithm.lower() == 'ddpg':
            from Algorithms.ddpg.core import CNNActorCritic            
            return CNNActorCritic
        elif RL_Algorithm.lower() == 'td3':
            from Algorithms.td3.core import CNNActorCritic            
            return CNNActorCritic
        elif RL_Algorithm.lower() == 'trpo':
            from Algorithms.trpo.core import CNNActorCritic            
            return CNNActorCritic
        elif RL_Algorithm.lower() == 'ppo':
            from Algorithms.ppo.core import CNNActorCritic            
            return CNNActorCritic
    
    raise AssertionError("Invalid model_type in config.json. Choose among ['mlp', 'cnn']")
