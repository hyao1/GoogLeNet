from torchvision import transforms
import os
from PIL import Image

from torch.utils.data import Dataset
import numpy as np
from collections import OrderedDict


class TorchDataset(Dataset):
    def __init__(self, root, class_num=200, img_size=448, is_train=True):
        '''
        :param root:下标(int)
        :return: data_len(int)
        '''
        # get image address
        self.train = is_train
        self.ImageAddress = []
        self.ImageLabel = []
        self.class_num = class_num

        # 读取测试集和训练集图片及标签
        self.classes = []

        if self.train:
            n = 0
            for class_name in sorted(os.listdir(root + '/train/')):
                self.classes.append(class_name)
                for image_name in os.listdir(root + '/train/' + class_name):
                    self.ImageAddress.append([root + '/train/' + class_name + '/' + image_name, image_name])
                    self.ImageLabel.append(n)
                n = n + 1

        else:
            n = 0
            for class_name in sorted(os.listdir(root + '/test/')):
                self.classes.append(class_name)
                for image_name in os.listdir(root + '/test/' + class_name):
                    self.ImageAddress.append([root + '/test/' + class_name + '/' + image_name, image_name])
                    self.ImageLabel.append(n)
                n = n + 1

        self.len = len(self.ImageAddress)
        print('%d个类, 总共%d个样本' % (class_num, len(self.ImageLabel)))

        if self.train:
            self.transforms = transforms.Compose(
                [transforms.Resize((600, 600)),
                 transforms.RandomCrop((img_size, img_size)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomRotation((0, 30), center=(224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )
        else:
            self.transforms = transforms.Compose(
                [transforms.Resize((600, 600)),
                 transforms.CenterCrop((img_size, img_size)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )

    def __getitem__(self, i):
        '''
        返回图片，图片名，类别
        :param path:下标(int)
        :return: data_len(int)
        '''
        index = i % self.len
        image_path, image_name = self.ImageAddress[index]
        label = self.ImageLabel[index]

        img = self.load_data(image_path)
        img = self.data_preproccess(img)
        label = np.array(label)
        return img, image_name, label

    def __len__(self):
        '''
       数据集大小
       :return: data_len(int)
       '''
        data_len = self.len
        return data_len

    def load_data(self, path):
        '''
        加载数据
        :param path:图片路径
        :return: image(numpy)
        '''
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:图片numpy数组
        :return:data(tensor)
        '''
        data = self.transforms(data)
        return data


if __name__ == "__main__":
    pass
