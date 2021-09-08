import linecache

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torchvision
import torchvision.transforms as tranforms

dataPath = '/home/xiaoshe/PycharmProjects/lfw/easy_trainingdata/'


class LFW(Dataset):
    def __init__(self, datapath, transform=None):
        self.path_dir = dataPath
        self.transform = transform
        self.images = []
        self.labels = []
        self.labels_onehot = []
        # load labels
        openmale = open(r'/home/xiaoshe/PycharmProjects/lfw/male_names.txt',
                        'rb')  # 打开文件并进行读
        data_man = openmale.read().decode('utf-8')
        openmale.close()
        openfemale = open(r'/home/xiaoshe/PycharmProjects/lfw/female_names.txt',
                          'rb')  # 打开文件并进行读
        data_woman = openfemale.read().decode('utf-8')
        openfemale.close()
        # 读取文件
        for i in range(int(self.dataCount // 2)):
            # 随机读取一行
            sum_male = data_man.count('\n')
            male_line = random.randint(1, sum_male + 1)
            male = linecache.getline(r'/home/xiaoshe/PycharmProjects/lfw/male_names.txt',
                                     male_line).strip()
            self.malelable[male] = 0
            sum_female = data_woman.count('\n')
            female_line = random.randint(1, sum_female + 1)
            female = linecache.getline(r'/home/xiaoshe/PycharmProjects/lfw/female_names.txt',
                                       female_line).strip()
            self.femalelabel[female] = 1
            print(male, female)
        # load pictures
        for key in self.malelable:
            picPath = self.dataPath + str(key)
            image = self.loadPictensor(picPath)
            label = 0
            self.images.append(image)
            self.labels.append(label)
        for key in self.femalelabel:
            picPath = self.dataPath + key
            image = self.loadPictensor(picPath)
            label = 1
            self.images.append(image)
            self.labels.append(label)

        self.labels_onehot = np.eye(2)[self.labels]

        # 打乱数据
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)
        np.random.set_state(state)
        np.random.shuffle(self.labels_onehot)

        # 按比例切割数据，分为训练集、验证集和测试集
        trainIndex = int(self.dataCount * 0.65)
        validationIndex = int(self.dataCount * 0.85)
        self.train_images = self.images[0: trainIndex]
        self.train_labels = self.labels[0: trainIndex]
        self.train_labels_onehot = self.labels_onehot[0: trainIndex]
        self.validation_images = self.images[trainIndex: validationIndex]
        self.validation_labels = self.labels[trainIndex: validationIndex]
        self.validation_labels_onehot = self.labels_onehot[trainIndex: validationIndex]
        self.test_images = self.images[validationIndex:]
        self.test_labels = self.labels[validationIndex:]
        self.test_labels_onehot = self.labels_onehot[validationIndex:]


    def loadPictensor(self, picFilePath):
        picData = Image.open(picFilePath)
        tensor_img = transforms.PILToTensor(picData)
        print(tensor_img)
        return tensor_img

print('Loading Lwf Data...')
dataLoader =DataLoader(dataPath)
dataLoader.loadLFW_label()
dataLoader.loadLFW_facedata()
