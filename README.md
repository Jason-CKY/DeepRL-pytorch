# Deep RL policies on Pybullet Environments

This repo is an implementation of various deep RL algorithm on pybullet robotic environments.

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
            <td> :x: </td>
            <td> :heavy_check_mark: </td>
        </tr>
        <tr>
            <td> PPO </td>
            <td> :heavy_check_mark: </td>
            <td> :heavy_check_mark: </td>
            <td> :x: </td>
            <td> :heavy_check_mark: </td>
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
            <td> Inverted Pendulum BulletEnv-v0 </td>
            <td> <img src = 'Model_Weights\InvertedPendulumBulletEnv-v0\comparison.png'> </td>
            <td><img src = 'Model_Weights\InvertedPendulumBulletEnv-v0\ppo\recording.gif'> </td>
        </tr>
        <tr>
            <td> Inverted Double Pendulum BulletEnv-v0 </td>
            <td> <img src = 'Model_Weights\InvertedDoublePendulumBulletEnv-v0\comparison.png'> </td>
            <td><img src = 'Model_Weights\InvertedDoublePendulumBulletEnv-v0\td3\recording.gif'> </td>
        </tr>
        <tr>
            <td> AntBulletEnv-v0 </td>
            <td> <img src = 'Model_Weights\AntBulletEnv-v0\comparison.png'> </td>
            <td><img src = 'Model_Weights\AntBulletEnv-v0\trpo\recording.gif'> </td>
        </tr>
        <tr>
            <td> HalfCheetahBulletEnv-v0 </td>
            <td> <img src = 'Model_Weights\HalfCheetahBulletEnv-v0\comparison.png'> </td>
            <td><img src = 'Model_Weights\HalfCheetahBulletEnv-v0\trpo\recording.gif'> </td>
        </tr>
    </tbody>
</table>

## Results comparing with Stable-Baselines3 agents

<table>
    <thead>
        <tr>
            <th>Environment</th>
            <th> Algorithm </th>
            <th> Learning Curve </th>
            <th> Episode Recording </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2> CartPole Continuous BulletEnv-v0 </td>
            <td> DDPG </td>
            <td> <img src = 'Model_Weights\CartPoleContinuousBulletEnv-v0\ddpg\comparison.png'> </td>
            <td><img src = 'Model_Weights\CartPoleContinuousBulletEnv-v0\ddpg\recording.gif'> </td>
        </tr>
        <tr>
            <td> TD3 </td>
            <td> <img src = 'Model_Weights\CartPoleContinuousBulletEnv-v0\td3\comparison.png'> </td>
            <td><img src = 'Model_Weights\CartPoleContinuousBulletEnv-v0\td3\recording.gif'> </td>
        </tr>
        <tr>
            <td rowspan=2> Inverted Pendulum BulletEnv-v0 </td>
            <td>DDPG</td>
            <td> <img src = 'Model_Weights\InvertedPendulumBulletEnv-v0\ddpg\comparison.png'> </td>
            <td><img src = 'Model_Weights\InvertedPendulumBulletEnv-v0\ddpg\recording.gif'></td>
        </tr>
        <tr>
            <td>TD3</td>
            <td> <img src = 'Model_Weights\InvertedPendulumBulletEnv-v0\td3\comparison.png'> </td>
            <td><img src = 'Model_Weights\InvertedPendulumBulletEnv-v0\td3\recording.gif'></td>
        </tr>
        <tr>
            <td rowspan=2> Inverted Double Pendulum BulletEnv-v0 </td>
            <td> DDPG </td>
            <td> <img src = 'Model_Weights\InvertedDoublePendulumBulletEnv-v0\ddpg\comparison.png'> </td>
            <td><img src = 'Model_Weights\InvertedDoublePendulumBulletEnv-v0\ddpg\recording.gif'></td>
        </tr>
        <tr>
            <td> TD3 </td>
            <td> <img src = 'Model_Weights\InvertedDoublePendulumBulletEnv-v0\td3\comparison.png'> </td>
            <td><img src = 'Model_Weights\InvertedDoublePendulumBulletEnv-v0\td3\recording.gif'></td>
        </tr>
        <tr>
            <td rowspan=2> AntBullet Env-v0 </td>
            <td> DDPG </td>
            <td><img src = 'Model_Weights\AntBulletEnv-v0\ddpg\comparison.png'></td>
            <td><img src = 'Model_Weights\AntBulletEnv-v0\ddpg\recording.gif'></td>
        </tr>
        <tr>
            <td> TD3 </td>
            <td><img src = 'Model_Weights\AntBulletEnv-v0\td3\comparison.png'></td>
            <td><img src = 'Model_Weights\AntBulletEnv-v0\td3\recording.gif'></td>
        </tr>
        <tr>
            <td rowspan=2> HalfCheetah BulletEnv-v0 </td>
            <td> DDPG </td>
            <td> <img src = 'Model_Weights\HalfCheetahBulletEnv-v0\ddpg\comparison.png'> </td>
            <td><img src = 'Model_Weights\HalfCheetahBulletEnv-v0\ddpg\recording.gif'></td>
        </tr>
        <tr>
            <td> TD3 </td>
            <td> <img src = 'Model_Weights\HalfCheetahBulletEnv-v0\td3\comparison.png'> </td>
            <td><img src = 'Model_Weights\HalfCheetahBulletEnv-v0\td3\recording.gif'></td>
        </tr>
    </tbody>
</table>

