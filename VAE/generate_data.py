import argparse
import gym
import pybullet_envs
import os
from PIL import Image
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, required=True, help='environment_id')
    parser.add_argument('--num_samples', type=int, required=True, 
                        help='specify number of image samples to generate') 
    parser.add_argument('--max_ep_len', type=int, default=100, help='Maximum length of an episode')
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
    parser.add_argument('--rlbench', action='store_true', help='if true, use rlbench environment wrappers')
    parser.add_argument('--view', type=str, default=None, 
                        choices=['wrist_rgb', 'front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb'], 
                        help='choose the type of camera view to generate image (only for RLBench envs)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.rlbench:
        assert args.view is not None
        import rlbench.gym
        from Wrappers.rlbench_wrapper import RLBench_Wrapper
        env = RLBench_Wrapper(gym.make(args.env), args.view)
    else:
        from Wrappers.image_learning import Image_Wrapper
        env = Image_Wrapper(gym.make(args.env))
    
    save_dir = os.path.join("dataset", args.env, args.view) if args.view is not None else os.path.join("dataset", args.env)
    os.makedirs(save_dir, exist_ok=True)

    image, ep_len = env.reset(), 0
    for i in tqdm(range(args.num_samples)):
        image, reward, done, _ = env.step(env.action_space.sample())
        ep_len += 1
        image = Image.fromarray(image)
        image.save(os.path.join(save_dir, f"image_{i}.png"))

        if done or (ep_len==args.max_ep_len):
            image, ep_len = env.reset(), 0



if __name__ == '__main__':
    main()