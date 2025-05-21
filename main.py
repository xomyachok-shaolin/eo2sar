import argparse
import os
import torch

from utils import check_folder, str2bool
# from UGATIT import UGATIT  # СТАРЫЙ ИМПОРТ
from SBBridge import SBGANTrainer  # НОВЫЙ ИМПОРТ

def parse_args():
    desc = "Train or Test SB-based MARM model"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train, test]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset name')
    parser.add_argument('--iteration', type=int, default=100000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='Log frequency')
    parser.add_argument('--save_freq', type=int, default=5000, help='Checkpoint save frequency')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--ch', type=int, default=64, help='Base channel number')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu','cuda:0','cuda:1'])
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Adversarial loss weight')
    parser.add_argument('--resume', type=bool, default=True)

    return parser.parse_args()

def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    # Создаём объект
    gan = SBGANTrainer(args)
    gan.build_model()

    if args.phase == 'train':
        gan.train()
    elif args.phase == 'test':
        gan.test()
    else:
        print("Unknown phase. Choose from [train, test].")

if __name__ == '__main__':
    main()
