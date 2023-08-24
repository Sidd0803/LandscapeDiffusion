
import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

from utils import *
from modules import UNet_conditional #,EMA

#best results achieved with 500 epochs, 32x32 images and 4000 training pictures
config = SimpleNamespace(    
    run_name = "DDPM_conditional",
    epochs = 500,
    noise_steps=1000,
    seed = 42,
    batch_size = 10,
    image_size = 32,
    num_classes = 5,
    dataset_path = r"dataConditional",
    train_folder = r"train_folder",
    val_folder = r"test_folder",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=2,
    lr = 5e-3)


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, image_size=256, num_classes=10, c_in=3, c_out=3, device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.image_size = image_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(device)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.image_size, self.image_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        print("shape of sampled output:", x.shape)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    """A function that tracks the progress of the diffusion training.
    A picture with 5 class colated is saved in the results folder"""
    def images_progress(self, epoch):
        labels = torch.arange(self.num_classes).long().to(self.device)
        sampled_images = self.sample(use_ema=False, labels=labels)
        save_images(sampled_images, os.path.join("results", f"progress",f"epoch{epoch}.jpg"))
        
        
    """A function that generates n photos for each class, storing them in separate folders"""
    def generate_n(self, n, model):
        labels = torch.arange(5).long().to("cuda")
        self.model = model
        for i in range(n):
            sampled_images = self.sample(use_ema=False, labels=labels)
            for j in labels:
                save_images(sampled_images[j], os.path.join("results", f"class{j+1}",f"picture{i+1}.jpg"))

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        "Save model locally"
        #torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.model, os.path.join("models", run_name, f"model_cond.pt"))
        #torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        #torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)
            
            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
            
            #log progress
            if epoch % args.log_every_epoch == 0:
                self.images_progress(epoch) 


        #save model
        self.save_model(run_name=args.run_name, epoch=epoch)
        



def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--image_size', type=int, default=config.image_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    set_seed(config.seed)

    #train the model
    diffuser = Diffusion(config.noise_steps, image_size=config.image_size, num_classes=config.num_classes)
    diffuser.prepare(config)
    diffuser.fit(config)
    
    #generate n images for each class
    model = torch.load("models/DDPM_conditional/model_cond.pt")
    model.eval() #for dropout and batchnorm
    diffuser.generate_n(30, model)