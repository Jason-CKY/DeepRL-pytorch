import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.Dataset import ImageDataset
from logger import Logger
from vae import VAE
from tqdm import tqdm
from PIL import Image

class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val

def save_plots(data, title, fpath):
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.plot(data)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.savefig(fpath)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default="dataset/reach_target-vision-v0/wrist_rgb", help='path to image folders')
    parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloaders')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--beta', type=float, default=1, help='Weighing value for KLD in B-VAE')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--save_freq', type=int, default=2000, help='save weights every <x> iterations')
    parser.add_argument('--log_freq', type=int, default=50, help='log losses every <x> iterations')
    parser.add_argument('--save_dir', type=str, default="output", help='path to save weights and logs')
    parser.add_argument('--load', type=str, default=None, help='path to load weights')
    
    return parser.parse_args()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.join(args.save_dir, "reconstruction"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "weights"), exist_ok=True)
    
    transform=transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    dataset = ImageDataset(args.dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Get 8 sample images to test the decoder on to visualise training progress
    sample_dataset = ImageDataset(args.dir, transform=val_transform)
    sample_dl = DataLoader(sample_dataset, batch_size=8, shuffle=True)

    vae = VAE(beta=args.beta).to(device)
    if args.load is not None:
        vae.load_weights(args.load)
    
    if args.ngpu>1:
        vae.dataparallel(args.ngpu)

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    logger = Logger(output_dir=args.save_dir)
    frames = []

    p_schedule = LinearSchedule(start=0.2, end=0.1, steps=args.epochs)

    for epoch in range(1, args.epochs+1):
        p = p_schedule()
        logger.running_log_dict = {
        'elbo': 0,
        'recon_loss': 0,
        'kl': 0
        }
        for i, image in enumerate(tqdm(dataloader), 1):
            vae.train()
            image = image.to(device)
            optimizer.zero_grad()
            elbo_loss, log_dict = vae.get_elbo_loss(image, p)
            elbo_loss.backward()
            optimizer.step()

            for k, v in log_dict.items():
                logger.running_log_dict[k] += v
            logger.store(**log_dict)

            if i%args.log_freq==0:
                print()
                print("#"* 50)
                print(f"p: {p}")
                print(f"Epoch: [{epoch}/{args.epochs}]\t Iter:[{i}/{len(dataloader)}]")
                for k, v in logger.running_log_dict.items():
                    print(f"{k}: {v/args.log_freq}")
                logger.running_log_dict = {
                'elbo': 0,
                'recon_loss': 0,
                'kl': 0
                }
                print("#"*50)        

            # Check how the model is doing by saving decoder's output on original input
            if (i % args.save_freq == 0) or ((epoch == args.epochs-1) and (i == len(dataloader)-1)):
                vae.eval()
                sample_images = next(iter(sample_dl)).to(device)
                with torch.no_grad():
                    im = vae(sample_images).cpu()
                im = torch.cat([sample_images.cpu(), im], dim=0)
                im = vutils.make_grid(im, padding=2).permute(1, 2, 0).numpy() * 0.5 + 0.5
                im = Image.fromarray((im*255).astype(np.uint8))
                im.save(os.path.join(args.save_dir, "reconstruction", f"epoch{epoch}_iter{i}.png"))
                frames.append(im)
                # vae.save_weights(os.path.join(args.save_dir, "weights", f"{epoch}_{i}.pth"))
                
        logger.dump()
    
    latest_weights_fname = '_'.join(args.dir.split('/')[1:]) + '.pth'
    vae.save_weights(os.path.join(args.save_dir, latest_weights_fname))
    frames[0].save(os.path.join(args.save_dir, 'animation.gif'), format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=0)

    elbo, kl, recon_loss = logger.load_results(['elbo', 'kl', 'recon_loss'])
    save_plots(elbo, title="ELBO loss curve", fpath=os.path.join(args.save_dir, "elbo_curve.png"))
    save_plots(kl, title="KL loss curve", fpath=os.path.join(args.save_dir, "kl_curve.png"))
    save_plots(recon_loss, title="reconstruction loss curve", fpath=os.path.join(args.save_dir, "recon_loss_curve.png"))


if __name__ == '__main__':
    main()