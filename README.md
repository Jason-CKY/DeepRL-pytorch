# Deep RL policies on Pybullet Environments

This repo is an implementation of various deep RL algorithm on pybullet robotic environments.

## Learning curves using Stable-Baselines3 agents on Pybullet Environments.

### HalfCheetahBulletEnv-v0

<table align='center'>
<tr align='center'>
<td> Algorithm </td>
<td> Learning Curve </td>
<td> Result </td>
</tr>
<tr>
<td> DDPG </td>
<td> <img src = 'Stable_Baselines\logs\HalfCheetahBulletEnv-v0\ddpg\learning_curve.png'> </td>
<td> <img src = 'Stable_Baselines\logs\HalfCheetahBulletEnv-v0\ddpg\recording.gif'>  </td>
</tr>
<tr>
<td> TD3 </td>
<td> <img src = 'Stable_Baselines\logs\HalfCheetahBulletEnv-v0\td3\learning_curve.png'> </td>
<td> <img src = 'Stable_Baselines\logs\HalfCheetahBulletEnv-v0\td3\recording.gif'>  </td>
</tr>
<tr>
<td> PPO </td><td> <img src = 'Stable_Baselines\logs\HalfCheetahBulletEnv-v0\ppo\learning_curve.png'> </td>
<td> <img src = 'Stable_Baselines\logs\HalfCheetahBulletEnv-v0\ppo\recording.gif'>  </td>
</tr>
</table>

### AntBulletEnv-v0

<table>
    <thead>
        <tr>
            <th>Layer 1</th>
            <th>Layer 2</th>
            <th>Layer 3</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>L1 Name</td>
            <td rowspan=2>L2 Name A</td>
            <td>L3 Name A</td>
        </tr>
        <tr>
            <td>L3 Name B</td>
        </tr>
        <tr>
            <td rowspan=2>L2 Name B</td>
            <td>L3 Name C</td>
        </tr>
        <tr>
            <td>L3 Name D</td>
        </tr>
    </tbody>
</table>

