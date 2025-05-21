# file: perceptual_loss.py
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Выделяем фичи из нескольких слоёв VGG19.
    Считаем L1/MSE между фичами генерированного и реального изображения.
    """
    def __init__(self, layers=('conv1_2','conv2_2','conv3_2','conv4_2'), use_l1=True):
        super(PerceptualLoss, self).__init__()
        self.vgg = _create_vgg19_extractor(layers)
        self.layers = layers
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

        # Замораживаем веса VGG
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # x, y: тензоры [N,3,H,W] в диапазоне [-1,1], нужно привести к [0,1] и потом нормализовать как для ImageNet
        x_feats = self.vgg(self._preprocess_input(x))
        y_feats = self.vgg(self._preprocess_input(y))
        loss = 0
        for layer in self.layers:
            loss += self.criterion(x_feats[layer], y_feats[layer])
        return loss

    def _preprocess_input(self, t):
        # [-1,1] -> [0,1]
        t = (t+1)/2
        # Нормализация под VGG (ImageNet-mean/std)
        mean = torch.tensor([0.485,0.456,0.406], device=t.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=t.device).view(1,3,1,1)
        t = (t - mean) / std
        return t

def _create_vgg19_extractor(layers):
    # Загружаем vgg19
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    # Пробежимся по слоям, сохраним нужные
    model = nn.Sequential()
    layer_map = {}
    current_name = ""
    output = {}

    name_list = {
        '0': 'conv1_1', 
        '1': 'relu1_1', 
        '2': 'conv1_2', 
        '3': 'relu1_2', 
        '4': 'pool1',
        '5': 'conv2_1', 
        '6': 'relu2_1', 
        '7': 'conv2_2', 
        '8': 'relu2_2', 
        '9': 'pool2',
        '10': 'conv3_1', 
        '11': 'relu3_1', 
        '12': 'conv3_2', 
        '13': 'relu3_2',
        '14': 'conv3_3', 
        '15': 'relu3_3', 
        '16': 'conv3_4', 
        '17': 'relu3_4', 
        '18': 'pool3',
        '19': 'conv4_1', 
        '20': 'relu4_1', 
        '21': 'conv4_2', 
        '22': 'relu4_2',
        '23': 'conv4_3', 
        '24': 'relu4_3', 
        '25': 'conv4_4', 
        '26': 'relu4_4', 
        '27': 'pool4',
        '28': 'conv5_1', 
        '29': 'relu5_1', 
        '30': 'conv5_2', 
        '31': 'relu5_2',
        '32': 'conv5_3', 
        '33': 'relu5_3', 
        '34': 'conv5_4', 
        '35': 'relu5_4', 
        '36': 'pool5'
    }

    # Превратим VGG-сеть в dict {layer_name -> output}, вернуть будем через forward hooks
    return VGG_FeatureExtractor(vgg, name_list, layers)

class VGG_FeatureExtractor(nn.Module):
    def __init__(self, vgg, name_list, layers_to_save):
        super().__init__()
        self.vgg = vgg
        self.name_list = name_list
        self.layers_to_save = set(layers_to_save)

        # Пробежимся и скопируем
        self.slice = nn.ModuleList()
        for i, layer in enumerate(vgg.children()):
            self.slice.append(layer)

    def forward(self, x):
        out = {}
        for i, layer in enumerate(self.slice):
            x = layer(x)
            layer_name = self.name_list[str(i)]
            if layer_name in self.layers_to_save:
                out[layer_name] = x
        return out
