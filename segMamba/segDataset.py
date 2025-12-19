import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image


def pathHelper(maskPath, prefix, imgType):
    idx = maskPath[-8: -4]
    if imgType == "enhanced":
        return os.path.join(prefix, 'enhancedImg', 'enhanced_' + idx + '.tif')
    elif imgType == "original":
        return os.path.join(prefix, 'Img', 'img_' + idx + '.tif')
    elif imgType == "both":
        return os.path.join(prefix, 'enhancedImg', 'enhanced_' + idx + '.tif'), os.path.join(prefix, 'Img', 'img_' + idx + '.tif')

class ImageDataset(Dataset):
    def __init__(self, dtype, device, imgType="enhanced"):
        self.dtype = dtype
        self.device = device
        self.mask =None
        self.toTensor = transforms.ToTensor()
        self.imgType = imgType
        if dtype == 'train':
            self.prefix = './dataSeg/train'
            valid_exts = ('.tif')
            mask_dir = os.path.join(self.prefix, 'mask')
            self.mask = sorted(
                [f for f in os.listdir(mask_dir) if f.lower().endswith(valid_exts) and not f.startswith('.')])
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])
        elif dtype == 'valid':
            self.prefix = './dataSeg/valid'
            valid_exts = ('.tif')
            mask_dir = os.path.join(self.prefix, 'mask')
            self.mask = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_exts) and not f.startswith('.')])
            self.transform = transforms.Compose([
                nn.Identity()
            ])
        else:
            raise ValueError('Unknown dataset')

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        maskPath = os.path.join(self.prefix, 'mask', self.mask[idx])
        imgPath = pathHelper(maskPath, self.prefix, self.imgType)
        if self.imgType != "both":
            img, mask = self.toTensor(Image.open(imgPath).convert('L')), self.toTensor(Image.open(maskPath).convert('L'))
            temp = torch.cat([img, mask], dim=0)
            transTemp = self.transform(temp)
            img, mask = torch.split(transTemp, 1, dim=0)
            return img.to(self.device), mask.to(self.device)
        elif self.imgType == "both":
            enhancedImg, img, mask = self.toTensor(Image.open(imgPath[0]).convert('L')), self.toTensor(Image.open(imgPath[1]).convert('L')), self.toTensor(Image.open(maskPath).convert('L'))
            temp = torch.cat([enhancedImg, img, mask], dim=0)
            transTemp = self.transform(temp)
            enhancedImg, img, mask = torch.split(transTemp, 1, dim=0)
            return enhancedImg.to(self.device), img.to(self.device), mask.to(self.device)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset('valid', device, imgType="both")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    max_mask, min_mask = -1, 1
    max_img, min_img = -1, 1
    for idx, batch in enumerate(loader):
        enhanced, img, mask = batch
        print(enhanced.shape, img.shape, mask.shape)