# timothy-Qcontrol-SAC-env
 RL_Zoo3 environment for training to be saved under classiccontrol and some instructions.

To replicate the results, do follow the steps below:
gymnasium:
1. Onto the local gymnasium package library 
    i.e. anaconda3/envs/{env_name}/Lib/site-packages/gymnasium/envs/classic_control
    , save the custom gymnasium environment in this repository (threelevle_Qcontrol.py)
2. Register the environment by writing
    'from gymnasium.envs.classic_control.threelevel_Qcontrol import threelevel_Qcontrol'
    under anaconda3/envs/{env_name}/Lib/site-packages/gymnasium/envs/classic_control/_init_.py
3. And finally register the environment under 
    anaconda3/envs/{env_name}/Lib/site-packages/gymnasium/envs/_init_.py by adding the following
    "register(
    id="threelevel_Qcontrol",
    entry_point="gymnasium.envs.classic_control.threelevel_Qcontrol:threelevel_Qcontrol",
    max_episode_steps=256,
    reward_threshold=100.0,
    kwargs={'Ωmax':20,
            'n_steps':256,
            'γ':5,
            'T':1,
            'reward_gain':1
            }
    )"
4. For more on registering custom environments, refer to https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym

rl_zoo3:
1. Onto the local gymnasium package library 
    i.e. anaconda3/envs/{env_name}/Lib/site-packages/rl_zoo3
    , replace train.py and exp_manager.py with the same files found in this repository. The files in this repository has edits that enable the use of wandb.
2. Under anaconda3/envs/{env_name}/Lib/site-packages/rl_zoo3/hyperparams/sac.yml
    add the following default parameters
    {threelevel_Qcontrol:
        n_timesteps: !!float 1.28e6
        policy: 'MlpPolicy'
        batch_size: 256
        learning_rate: lin_7.3e-4
        buffer_size: 1000000
        ent_coef: 'auto_0.005'
        gamma: 0.95
        tau: 0.01
        train_freq: 1
        gradient_steps: 1
        learning_starts: 10000
        policy_kwargs: "dict(activation_fn=th.nn.Tanh,net_arch=dict(pi=[256,256,256], qf=[64, 64]))"}
3. For more information on training and hyperparameter tuning with rl_zoo3, refer to https://rl-baselines3-zoo.readthedocs.io/en/master/index.html