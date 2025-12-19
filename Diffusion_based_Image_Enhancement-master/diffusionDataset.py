import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image


def pathHelper(DSImgPath, prefix, maskFlag):
    idx = DSImgPath[-8: -4]
    if maskFlag:
        return (os.path.join(prefix, 'groundTruth', 'img_' + idx + '.tif'),
                os.path.join(prefix, 'mask', 'mask_' + idx + '.tif'))
    else:
        return os.path.join(prefix, 'groundTruth', 'img_' + idx + '.tif')

class ImageDataset(Dataset):
    def __init__(self, dtype, device, maskFlag=False):
        self.dtype = dtype
        self.device = device
        self.data =None
        self.toTensor = transforms.ToTensor()
        self.maskFlag = maskFlag
        if dtype == 'train':
            self.prefix = './data/train'
            valid_exts = ('.tif')
            img_dir = os.path.join(self.prefix, 'DSImg')
            self.data = sorted(
                [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts) and not f.startswith('.')])
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                #transforms.RandomAffine(25, translate=(0.1, 0.1)),
            ])
        elif dtype == 'valid':
            self.prefix = './data/valid'
            valid_exts = ('.tif')
            img_dir = os.path.join(self.prefix, 'DSImg')
            self.data = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts) and not f.startswith('.')])
            self.transform = transforms.Compose([
                nn.Identity()
            ])
        else:
            raise ValueError('Unknown dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        DSImgPath = os.path.join(self.prefix, 'DSImg', self.data[idx])
        if self.maskFlag:
            groundTruthPath, maskPath = pathHelper(DSImgPath, self.prefix, self.maskFlag)
            DSImg, truth, imgMask = (self.toTensor(Image.open(DSImgPath).convert('L')),
                                     self.toTensor(Image.open(groundTruthPath).convert('L')),
                                     self.toTensor(Image.open(maskPath).convert('L')))
            temp = torch.cat([DSImg, truth, imgMask], dim=0)
            transTemp = self.transform(temp)
            DSImg, truth, imgMask = torch.split(transTemp, 1, dim=0)
            return DSImg, truth, imgMask
        else:
            groundTruthPath = pathHelper(DSImgPath, self.prefix, self.maskFlag)
            DSImg, truth = (self.toTensor(Image.open(DSImgPath).convert('L')),
                            self.toTensor(Image.open(groundTruthPath).convert('L')))
            temp = torch.cat([DSImg, truth], dim=0)
            transTemp = self.transform(temp)
            DSImg, truth = torch.split(transTemp, 1, dim=0)
            return DSImg, truth

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset('valid', device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, batch in enumerate(loader):
        DSImg, truth = batch
        print(DSImg.shape, truth.shape)
