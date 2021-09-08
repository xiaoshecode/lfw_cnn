import linecache
from torchvision.transforms import transforms
import random
# 导入自定义的数据加载包
import linecache
import random
from scipy.special import expit
import numpy as np
from PIL import Image
import operator


class DataLoader:
    images = []
    labels = []
    labels_onehot = []
    # 现路径为本机上的路径，若加载不了图片，可考虑路径问题
    dataPath = '/home/xiaoshe/PycharmProjects/lfw/easy_trainingdata/'
    # 训练集列表
    train_images = []
    train_labels = []
    train_labels_onehot = []
    # 验证集列表
    validation_images = []
    validation_labels = []
    validation_labels_onehot = []
    # 测试集列表
    test_images = []
    test_labels = []
    test_labels_onehot = []
    # 加载图片数量，可修改,是偶数
    dataCount = 500

    # 标签字典
    malelable = {}
    femalelabel = {}

    def loadPictensor(self, picFilePath):
        picData = Image.open(picFilePath)
        tensor_img = transforms.PILToTensor(picData)
        print(tensor_img)
        return tensor_img

    # 加载标签数据
    def loadLFW_label(self):
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
            print(male,female)

    def print_lable(self):
        for key in self.malelable:
            print("key:" + key + "value:", self.malelable[key])
        for key in self.femalelabel:
            print("key:" + key + "value:", self.femalelabel[key])

    # 加载图片数据
    def loadLFW_facedata(self):
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


    # traindata
    def getTrainData(self):
        return self.train_images, self.train_labels, self.train_labels_onehot

    # 验证集
    def getValidationData(self):
        return self.validation_images, self.validation_labels, self.validation_labels_onehot

    # 测试集
    def getTestData(self):
        return self.test_images, self.test_labels, self.test_labels_onehot



    print('Loading Lwf Data...')
    dataLoader =DataLoader()
    dataLoader.loadLFW_label()
    dataLoader.loadLFW_facedata()
