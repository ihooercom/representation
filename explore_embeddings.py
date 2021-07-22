#%%
import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
import os
import random
import numpy as np

from torchvision.datasets.folder import default_loader

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class SmallImageNetDataset(VisionDataset):
    def __init__(self, root: str, n: int, transform) -> None:
        self.root = root
        self.n = n
        super(SmallImageNetDataset, self).__init__(root=root, transform=transform)
        self.samples = self.load_imagenet_imglist(self.root)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, int(target)

    def __len__(self) -> int:
        return len(self.samples)

    def load_imagenet_imglist(self, root_dir):
        val_path = os.path.join(root_dir, 'meta/val.txt')
        with open(val_path, 'r') as f:
            lines = f.readlines()
        lst = []
        samples = random.sample(lines, self.n)
        for sample in samples:
            image_name, label = sample.strip().split()
            imgpath = os.path.join(root_dir, 'val', image_name)
            lst.append((imgpath, label))
        return lst


def extract_features(model_entry, model_name):
    model = torch.hub.load(model_entry, model_name)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SmallImageNetDataset(root='/workspace/data/dataset/cv/imagenet/images/', n=1000, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    model.to('cuda')

    feats = []
    probs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(dataloader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            feats.append(model.feat)

            probabilities = torch.nn.functional.softmax(output, dim=1)
            probs.append(probabilities)

    feats = torch.cat(feats, dim=0).detach().cpu().numpy()
    probs = torch.cat(probs, dim=0).detach().cpu().numpy()
    if hasattr(model, 'classifier'):
        weight = model.classifier.weight.detach().cpu().numpy()
        bias = model.classifier.bias.detach().cpu().numpy()
    elif hasattr(model, 'fc'):
        weight = model.fc.weight.detach().cpu().numpy()
        weight = model.fc.weight.detach().cpu().numpy()
    else:
        raise NotImplementedError
    torch.save({'feats': feats, 'probs': probs, 'weight': weight, 'bias': bias}, f'feature-{model_name}.th')


#%%
# extract_features('pytorch/vision:v0.9.0', 'densenet121')
# extract_features('pytorch/vision:v0.9.0', 'resnext50_32x4d')
# extract_features('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
#%%
fd = torch.load('feature-densenet121.th')
ft = fd['feats']
pb = fd['probs']
w = fd['weight']
b = fd['bias']
print(ft.shape)
print(w.shape)

w_list = w.reshape(1, -1)[0]
print(w_list.shape)
import matplotlib.pyplot as plt

plt.hist(w_list, bins=1000)
plt.show()


#%%
fd = torch.load('feature-resnext101_32x8d_wsl.th')
ft = fd['feats']
pb = fd['probs']
w = fd['weight']
b = fd['bias']
print(ft.shape)
print(w.shape)

w_list = w.reshape(1, -1)[0]
print(w_list.shape)
import matplotlib.pyplot as plt

plt.hist(w_list, bins=1000)
plt.show()


# %%
