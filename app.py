import os
import argparse
import torch
import cv2
from PIL import Image
from torchvision import transforms

from SBBridge import SBGANTrainer  # Импортируем наш класс

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='results/YOUR_DATASET_NAME/model/SB_0100000.pt')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu','cuda:0','cuda:1'])
    parser.add_argument('--img', type=str, default='test_eo.tif', help='Path to input EO image')
    parser.add_argument('--out_img', type=str, default='result_sar.png', help='Path to output SAR image')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--ch', type=int, default=64)
    return parser.parse_args()

class SBGANInference:
    def __init__(self, model_path, device='cuda:0', img_size=256, ch=64):
        self.device = device
        self.img_size = img_size
        self.ch = ch
        # создаём "пустой" объект тренера
        self.trainer = SBGANTrainer(argparse.Namespace(
            dataset='',
            result_dir='',
            batch_size=1,
            iteration=1,
            print_freq=100,
            save_freq=1000,
            lr=0.0002,
            weight_decay=0.0,
            img_size=self.img_size,
            device=self.device,
            ch=self.ch
        ))
        self.trainer.build_model()
        self.trainer.load(model_path)
        self.trainer.gen.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def run_inference(self, img_path, out_path):
        # Загружаем изображение
        img_pil = Image.open(img_path).convert('RGB')
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fake_sar = self.trainer.gen(input_tensor)
        # Преобразуем в numpy
        fake_sar_np = fake_sar[0].detach().cpu().permute(1,2,0).numpy()
        # denorm: из [-1,1] -> [0,1]
        fake_sar_np = (fake_sar_np * 0.5) + 0.5
        fake_sar_np = (fake_sar_np * 255.0).astype('uint8')
        fake_sar_bgr = cv2.cvtColor(fake_sar_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, fake_sar_bgr)
        print(f"Saved inference result to {out_path}")

def main():
    args = parse_args()
    infer = SBGANInference(
        model_path=args.model_path,
        device=args.device,
        img_size=args.img_size,
        ch=args.ch
    )
    infer.run_inference(args.img, args.out_img)

if __name__ == '__main__':
    main()
