import torch
from torch import nn
import cv2
import numpy as np
from Resnet18 import Resnet18
from torchvision import transforms, models
from MyDataset import MyDataset
from matplotlib import pyplot as plt

feature_maps = []

def _get_feature_hook(module, input, out):
    feature_maps.append(out)


def plot_feature_map(features):
    print(features.shape)
    features -= np.min(features, axis=1, keepdims=True)
    features /= np.max(features, axis=1, keepdims=True)
    imgs = np.uint8(features * 255)
    imgs = imgs.squeeze(0)
    # imgs = np.expand_dims(imgs, axis=-1)
    for i in range(imgs.shape[0]):
        plt.plot(8, 8, i)
        plt.imshow(imgs[i], cmap=plt.cm.jet)
        print(imgs[i].shape)
    plt.colorbar()
    plt.show()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

dataset = MyDataset('./train')

net = models.resnet18()
net.fc = nn.Linear(512, 2)
net.load_state_dict(torch.load("./resnet18_2.pkl"))
net.layer4.register_forward_hook(_get_feature_hook)
# print(net)

input, _ = dataset[0]
# plt.imshow(input)
# plt.show()
# exit(0)
input = transform(input)
input = input.unsqueeze(0)
out = net(input)
plot_feature_map(feature_maps[0].detach().numpy())


