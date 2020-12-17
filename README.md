# Deep RL policies on Pybullet Environments

This repo is a pytorch implementation of various deep RL algorithms, trained and evaluated on pybullet robotic environments.

## Dependencies:
* torch==1.6.0
* torchvision==0.7.0
* CUDA >= 10.2
* RLBench

## Implemented Algorithms:

<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Discrete actions</th>
            <th>Continuous actions</th>
            <th>Stochastic policy</th>
            <th>Deterministic policy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> DDPG </td>
            <td> :x: </td>
            <td> :heavy_check_mark: </td>
            <td> :x: </td>
            <td> :heavy_check_mark: </td>
        </tr>
        <tr>
            <td> TD3 </td>
            <td> :x: </td>
            <td> :heavy_check_mark: </td>
            <td> :x: </td>
            <td> :heavy_check_mark: </td>
        </tr>
         <tr>
            <td> TRPO </td>
            <td> :heavy_check_mark: </td>
            <td> :heavy_check_mark: </td>
            <td> :heavy_check_mark: </td>
            <td> :x: </td>
        </tr>
        <tr>
            <td> PPO </td>
            <td> :heavy_check_mark: </td>
            <td> :heavy_check_mark: </td>
            <td> :heavy_check_mark: </td>
            <td> :x: </td>
        </tr>       
    </tbody>
</table>


## Comparison of results in PyBullet Environments
<table>
    <thead>
        <tr>
            <th>Environment</th>
            <th> Learning Curve </th>
            <th> Episode Recording </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> CartPole Continuous BulletEnv-v0 </td>
            <td> <img src = 'Model_Weights\CartPoleContinuousBulletEnv-v0\comparison.png'> </td>
            <td><img src = 'Model_Weights\CartPoleContinuousBulletEnv-v0\ddpg\recording.gif'> </td>
        </tr>
        <tr>
            <td> Hopper BulletEnv-v0 </td>
            <td> <img src = 'Model_Weights\HopperBulletEnv-v0\comparison.png'> </td>
            <td><img src = 'Model_Weights\HopperBulletEnv-v0\td3\recording.gif'> </td>
        </tr>
        <tr>
            <td> AntBulletEnv-v0 </td>
            <td> <img src = 'Model_Weights\AntBulletEnv-v0\comparison.png'> </td>
            <td><img src = 'Model_Weights\AntBulletEnv-v0\td3\recording.gif'> </td>
        </tr>
        <tr>
            <td> HalfCheetahBulletEnv-v0 </td>
            <td> <img src = 'Model_Weights\HalfCheetahBulletEnv-v0\comparison.png'> </td>
            <td><img src = 'Model_Weights\HalfCheetahBulletEnv-v0\ddpg\recording.gif'> </td>
        </tr>
    </tbody>
</table>

## How to use
* Clone this repo
* pip install -r requirements.txt

### Training model for openai gym environment
* Edit training parameters in ./Algorithms/<algo>/config.json
```
python train.py
usage: train.py [-h] [--env ENV] [--agent AGENT] --timesteps TIMESTEPS
                [--seed SEED] [--num_trials NUM_TRIALS]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment_id
  --agent AGENT         specify type of agent (e.g. DDPG/TRPO/PPO/random)
  --timesteps TIMESTEPS
                        specify number of timesteps to train for
  --seed SEED           seed number for reproducibility
  --num_trials NUM_TRIALS
                        Number of times to train the algo
```

### Testing trained model performance
```
python test.py
usage: test.py [-h] [--env ENV] [--agent AGENT] [--render] [--gif]
               [--timesteps TIMESTEPS] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment_id
  --agent AGENT         specify type of agent (e.g. DDPG/TRPO/PPO/random)
  --render              if true, display human renders of the environment
  --gif                 if true, make gif of the trained agent
  --timesteps TIMESTEPS
                        specify number of timesteps to train for
  --seed SEED           seed number for reproducibility
```