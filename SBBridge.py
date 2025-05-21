# file: SBBridge.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils as vutils

import numpy as np
import cv2

from dataset import ImageFolder
from utils import check_folder, RGB2BGR, denorm, tensor2numpy

# Импортируем наш генератор и дискриминатор, а также перцептуальный лосс
from hierarchical_marm_generator import HierarchicalMARMGenerator
from attn_discriminator import AttnDiscriminator
from perceptual_loss import PerceptualLoss

from torch.utils.tensorboard import SummaryWriter

def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

class SBGANTrainer(object):
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.result_dir = args.result_dir
        self.batch_size = args.batch_size
        self.iteration = args.iteration
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.img_size = args.img_size

        # Здесь device может быть "cuda:0" или "cpu", но при DataParallel важнее,
        # что мы просто используем cuda вообще.
        self.device = args.device

        self.ch = args.ch

        self.adv_weight = getattr(args, 'adv_weight', 1.0)
        self.sb_weight = getattr(args, 'sb_weight', 1.0)
        self.percep_weight = getattr(args, 'percep_weight', 1.0)

        self.resume = getattr(args, 'resume', False)

        # TensorBoard
        log_dir = os.path.join(self.result_dir, self.dataset, 'logs')
        check_folder(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

        print("===== SBGANTrainer: INIT =====")
        print(f"Using device: {self.device}, batch_size={self.batch_size}, iteration={self.iteration}")

    def build_model(self):
        print("===== BUILD MODEL =====")

        # 1) Dataset + Dataloader
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        trainA_path = os.path.join(self.dataset, 'trainA')
        trainB_path = os.path.join(self.dataset, 'trainB')

        self.trainA = ImageFolder(trainA_path, transform=train_transform)
        self.trainB = ImageFolder(trainB_path, transform=train_transform)

        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # 2) Создаём модели (на 1й GPU или cpu)
        netG = HierarchicalMARMGenerator(in_ch=3, out_ch=3, base_ch=self.ch)
        netD = AttnDiscriminator(in_ch=3, base_ch=self.ch)

        netG.to(self.device)
        netD.to(self.device)

        # 3) DataParallel, если у нас > 1 GPU
        ngpu = torch.cuda.device_count()
        print(f"Number of GPUs detected: {ngpu}")
        if ngpu > 1:
            # например, используем 6 (или min(6,ngpu)) GPU
            used_gpus = list(range(min(6, ngpu)))
            print(f"Using GPUs: {used_gpus}")
            netG = nn.DataParallel(netG, device_ids=used_gpus)
            netD = nn.DataParallel(netD, device_ids=used_gpus)

        self.gen = netG
        self.dis = netD

        # 4) Лоссы
        self.L1_loss = nn.L1Loss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.perceptual_loss = PerceptualLoss(
            layers=('conv1_2','conv2_2','conv3_2'),
            use_l1=True
        ).to(self.device)

        # 5) Оптимизаторы
        self.G_optim = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.5,0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(self.dis.parameters(), lr=self.lr, betas=(0.5,0.999), weight_decay=self.weight_decay)

        self.start_iter = 1
        if self.resume:
            self.start_iter = self.load_latest_ckpt()

        print("===== MODEL BUILT SUCCESSFULLY =====")

    def compute_adv_loss(self, pred, target_is_real):
        labels = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.BCE_loss(pred, labels)

    def compute_sb_loss(self, fake_B, real_B):
        # Заглушка: L1
        return self.L1_loss(fake_B, real_B)

    def train(self):
        print("===== START TRAINING =====")

        dataA_iter = iter(self.trainA_loader)
        dataB_iter = iter(self.trainB_loader)
        start_time = time.time()

        for step in range(self.start_iter, self.iteration+1):
            try:
                real_A, _ = next(dataA_iter)
            except StopIteration:
                dataA_iter = iter(self.trainA_loader)
                real_A, _ = next(dataA_iter)

            try:
                real_B, _ = next(dataB_iter)
            except StopIteration:
                dataB_iter = iter(self.trainB_loader)
                real_B, _ = next(dataB_iter)

            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            # === Update Dis ===
            self.D_optim.zero_grad()
            fake_B = self.gen(real_A)
            pred_real_B = self.dis(real_B)
            pred_fake_B = self.dis(fake_B.detach())

            D_loss_real = self.compute_adv_loss(pred_real_B, True)
            D_loss_fake = self.compute_adv_loss(pred_fake_B, False)
            D_loss = 0.5*(D_loss_real + D_loss_fake)
            D_loss.backward()
            self.D_optim.step()

            # === Update Gen ===
            self.G_optim.zero_grad()
            fake_B = self.gen(real_A)
            pred_fake_B2 = self.dis(fake_B)

            G_adv_loss = self.compute_adv_loss(pred_fake_B2, True)*self.adv_weight
            sb_loss_val = self.compute_sb_loss(fake_B, real_B)*self.sb_weight
            percep_loss_val = self.perceptual_loss(fake_B, real_B)*self.percep_weight

            G_loss = G_adv_loss + sb_loss_val + percep_loss_val
            G_loss.backward()
            self.G_optim.step()

            # === Logs & Save ===
            if step % self.print_freq == 0:
                elapsed = time.time() - start_time
                print(f"[{step}/{self.iteration}] time:{elapsed:.2f}, D:{D_loss.item():.4f}, G:{G_loss.item():.4f}")
                start_time = time.time()

                # TensorBoard
                self.writer.add_scalar("Loss/D", D_loss.item(), step)
                self.writer.add_scalar("Loss/G", G_loss.item(), step)
                self.writer.add_scalar("Loss/G_adv", G_adv_loss.item(), step)
                self.writer.add_scalar("Loss/SB", sb_loss_val.item(), step)
                self.writer.add_scalar("Loss/Percep", percep_loss_val.item(), step)

                # Пример изображений
                with torch.no_grad():
                    fake_B2A = self.gen(fake_B)
                disp = torch.cat([ (real_A*0.5+0.5), (fake_B*0.5+0.5), (fake_B2A*0.5+0.5), (real_B*0.5+0.5) ], dim=0)
                grid = vutils.make_grid(disp, nrow=self.batch_size, normalize=False)
                self.writer.add_image('Train/R_A_F_B_F_B2A_R_B', grid, step)

                # Пример PSNR
                fake_0 = (fake_B[0].detach().cpu().numpy()*0.5)+0.5
                real_0 = (real_B[0].detach().cpu().numpy()*0.5)+0.5
                fake_0 = np.transpose(fake_0, (1,2,0))
                real_0 = np.transpose(real_0, (1,2,0))
                psnr_val = compute_psnr(fake_0, real_0)
                self.writer.add_scalar("Metric/PSNR", psnr_val, step)

            if step % self.save_freq == 0:
                self.save(step)

        self.writer.close()
        print("===== TRAINING FINISHED =====")

    def save(self, step):
        model_dir = os.path.join(self.result_dir, self.dataset, 'model')
        check_folder(model_dir)
        # Если мы в DataParallel, state_dict() содержит module. Но PyTorch 1.7+ обычно сам может это учесть.
        # Однако, иногда делают: self.gen.module.state_dict() и self.dis.module.state_dict() если DP
        # Ниже используем простую схему (см. load)
        params = {
            'gen': self.gen.state_dict(),
            'dis': self.dis.state_dict(),
            'g_optim': self.G_optim.state_dict(),
            'd_optim': self.D_optim.state_dict(),
            'iter': step
        }
        torch.save(params, os.path.join(model_dir, f"SB_{step:07d}.pt"))
        print(f"Saved model checkpoints at step {step}")

    def load_latest_ckpt(self):
        model_dir = os.path.join(self.result_dir, self.dataset, 'model')
        if not os.path.exists(model_dir):
            return 1
        ckpts = [f for f in os.listdir(model_dir) if f.startswith('SB_') and f.endswith('.pt')]
        if len(ckpts) == 0:
            return 1
        ckpts.sort()
        last_ckpt = ckpts[-1]
        full_path = os.path.join(model_dir, last_ckpt)
        self.load(full_path)
        return self.start_iter

    def load(self, ckpt_path):
        params = torch.load(ckpt_path, map_location=self.device)
        self.gen.load_state_dict(params['gen'])
        self.dis.load_state_dict(params['dis'])
        self.G_optim.load_state_dict(params['g_optim'])
        self.D_optim.load_state_dict(params['d_optim'])
        self.start_iter = params.get('iter', 1)
        print(f"[*] Loaded from {ckpt_path}, resume at iter={self.start_iter}")

    def test(self):
        print("===== START TESTING =====")

        testA_path = os.path.join(self.dataset, 'testA')
        transform_test = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        testA_dataset = ImageFolder(testA_path, transform=transform_test)
        testA_loader = DataLoader(testA_dataset, batch_size=1, shuffle=False)

        result_test_dir = os.path.join(self.result_dir, self.dataset, 'test')
        check_folder(result_test_dir)

        self.gen.eval()

        with torch.no_grad():
            for i, (real_A,_) in enumerate(testA_loader):
                real_A = real_A.to(self.device)
                fake_B = self.gen(real_A)
                fake_B_np = tensor2numpy(denorm(fake_B[0]))
                fake_B_np = RGB2BGR(fake_B_np)
                cv2.imwrite(os.path.join(result_test_dir,f"A2B_{i+1}.png"), fake_B_np*255.0)
        print("===== TEST FINISHED =====")
