import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Импортируем классы и функции из ваших файлов
from networks import ResnetGenerator
from utils import denorm, tensor2numpy, RGB2BGR, cam

# --------------------- Параметры --------------------- #
device = 'cuda:0'                       # или 'cpu', если нет GPU
model_path = 'eo2sar_params_0137000.pt' # путь к предобученной модели
image_dir = 'SN6-SAROPT/valAA'  # путь к папке с оптическими изображениями
sar_dir = 'SN6-SAROPT/valBB'
output_dir = 'translated_results'       # куда сохранять результаты
img_size = 256
ch = 64
n_res = 4
light = True  # если модель была обучена в "light" режиме, ставим True; если нет – False
# ---------------------------------------------------- #

os.makedirs(output_dir, exist_ok=True)

# Трансформация, аналогичная test_transform
transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Загрузка генераторов (сетки A2B и B2A)
genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch, n_blocks=n_res,
                         img_size=img_size, light=light).to(device)
genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=ch, n_blocks=n_res,
                         img_size=img_size, light=light).to(device)

# Загрузка весов
print(f"Загружаю модель из {model_path} ...")
params = torch.load(model_path, map_location=device)
genA2B.load_state_dict(params['genA2B'])
genB2A.load_state_dict(params['genB2A'])

genA2B.eval()
genB2A.eval()

# Собираем список файлов изображений в директории
image_paths = sorted(glob.glob(os.path.join(image_dir, '*.*')))
if not image_paths:
    print(f"Не найдено изображений в {image_dir}")
    exit(0)

print(f"Найдено {len(image_paths)} изображений. Начинаю обработку...\n")

for idx, img_path in enumerate(image_paths, start=1):
    base_name = os.path.basename(img_path)
    # 8-м шагом добавляем реальное SAR, ищем файл в sar_dir
    real_sar_path = os.path.join(sar_dir, 'SAR' + base_name[3:])

    if not os.path.isfile(real_sar_path):
        print(f"[{idx}] Нет соответствующего SAR-файла для {base_name}, пропускаю.")
        continue

    # 1. Чтение и препроцессинг изображения
    img_pil = Image.open(img_path).convert('RGB')
    real_A = transform_test(img_pil).unsqueeze(0).to(device)  # (1,3,H,W)


    # ----- 2) Чтение реального SAR -----
    sar_pil = Image.open(real_sar_path).convert('RGB')
    real_B = transform_test(sar_pil).unsqueeze(0).to(device)

    # 2. Прогон через генераторы

    # Генерируем: A -> B
    fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap = genA2B(real_A)
    # Генерируем: B -> A, но здесь нам нужно на вход передать fake_A2B
    fake_A2B2A, _, fake_A2B2A_heatmap = genB2A(fake_A2B)
    # Параллельно генерируем A->A (identity) при помощи genB2A
    fake_A2A, _, fake_A2A_heatmap = genB2A(real_A)

    # 3. Извлекаем тензоры в np.array
    #    Для удобства используем denorm -> tensor2numpy -> RGB2BGR
    #    + для тепловых карт (heatmap) используем функцию cam()

    realA_np     = RGB2BGR(tensor2numpy(denorm(real_A[0])))
    fakeA2A_np   = RGB2BGR(tensor2numpy(denorm(fake_A2A[0])))
    fakeA2B_np   = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
    fakeA2B2A_np = RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))

    # Тепловые карты
    fakeA2A_cam   = cam(tensor2numpy(fake_A2A_heatmap[0]), size=img_size)
    fakeA2B_cam   = cam(tensor2numpy(fake_A2B_heatmap[0]), size=img_size)
    fakeA2B2A_cam = cam(tensor2numpy(fake_A2B2A_heatmap[0]), size=img_size)

    realB_np = RGB2BGR(tensor2numpy(denorm(real_B[0])))

    # 4. Склеиваем в один ряд (ось=0 значит вертикально «снизу-вверх», а нам нужно горизонтально — поэтому axis=1)
    #    Но здесь в коде UGATIT стоит np.concatenate(..., 0), поскольку там структура уже (H, W, 3). 
    #    Для «в линию» передаём axis=1, если хотим одну длинную «полосу».
    #    В оригинальном коде UGATIT как раз axis=0 означает "по вертикали", 
    #    т.к. используется A2B = np.concatenate((...), 0)
    #    но дальше A2B * 255.0 пишет изображение как матрицу высоты H, ширины 7*W.
    #    
    #    С учётом что всё уже в формате (H, W, C),
    #    np.concatenate([...], axis=0) — склейка сверху вниз,
    #    np.concatenate([...], axis=1) — слева направо.
    #    
    #    Ниже я делаю «в одной строке» (как в UGATIT) — axis=0:

    # Внимание: чтобы картинки шли «горизонтально» — меняем axis=1:
    # Для 7 кадров будем склеивать слева направо, значит axis=1.
    concat_img = np.concatenate((
        realA_np,              # (1) Real optical
        fakeA2A_cam,           # (2) Heat map A->A
        fakeA2A_np,            # (3) Fake optical
        fakeA2B_cam,           # (4) Heat map A->B
        fakeA2B_np,            # (5) Fake SAR
        fakeA2B2A_cam,         # (6) Heat map B->A (после A->B)
        fakeA2B2A_np,           # (7) Fake optical
        realB_np            # (8) Real SAR
    ), axis=1)

    # 5. Сохраняем результат
    #    Домножаем на 255, т.к. в OpenCV пиксели должны быть в диапазоне [0, 255]
    filename = os.path.basename(img_path)
    out_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_translated.png')
    cv2.imwrite(out_path, concat_img * 255.0)

    print(f"[{idx}/{len(image_paths)}] Сохранён результат: {out_path}")

print("\nОбработка завершена!")
