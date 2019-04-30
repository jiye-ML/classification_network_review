import os
import shutil
import zipfile
import numpy as np
from glob import glob
import scipy.misc as misc
import pickle
import matplotlib.pyplot as plt

from src.Tools import Tools


class PreData:
    '''
    如果使用resis数据集， 需要解压文件
    '''

    def __init__(self, zip_file, ratio=4):
        data_path = zip_file.split(".zip")[0]
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")

        if not os.path.exists(data_path):
            f = zipfile.ZipFile(zip_file, "r")
            f.extractall(data_path)

            all_image = self.get_all_images(os.path.join(data_path, data_path.split("/")[-1]))
            self.get_data_result(all_image, ratio, Tools.new_dir(self.train_path), Tools.new_dir(self.test_path))
        else:
            Tools.print_info("data is exists")
        pass

    # 生成测试集和训练集
    @staticmethod
    def get_data_result(all_image, ratio, train_path, test_path):
        train_list = []
        test_list = []

        # 遍历
        Tools.print_info("bian")
        for now_type in range(len(all_image)):
            now_images = all_image[now_type]
            for now_image in now_images:
                # 划分
                if np.random.randint(0, ratio) == 0:  # 测试数据
                    test_list.append((now_type, now_image))
                else:
                    train_list.append((now_type, now_image))
            pass

        # 打乱
        Tools.print_info("shuffle")
        np.random.shuffle(train_list)
        np.random.shuffle(test_list)

        # 提取训练图片和标签
        Tools.print_info("train")
        for index in range(len(train_list)):
            now_type, image = train_list[index]
            shutil.copyfile(image, os.path.join(train_path,
                                                str(np.random.randint(0, 1000000)) + "-" + str(now_type) + ".jpg"))

        # 提取测试图片和标签
        Tools.print_info("test")
        for index in range(len(test_list)):
            now_type, image = test_list[index]
            shutil.copyfile(image, os.path.join(test_path,
                                                str(np.random.randint(0, 1000000)) + "-" + str(now_type) + ".jpg"))

        pass

    # 所有的图片
    @staticmethod
    def get_all_images(images_path):
        all_image = []
        all_path = os.listdir(images_path)
        for one_type_path in all_path:
            now_path = os.path.join(images_path, one_type_path)
            if os.path.isdir(now_path):
                now_images = glob(os.path.join(now_path, '*.jpg'))
                all_image.append(now_images)
            pass
        return all_image

    # 生成数据
    @staticmethod
    def main(zip_file):
        pre_data = PreData(zip_file)
        return pre_data.train_path, pre_data.test_path

    pass


class Data:
    def __init__(self, batch_size, type_number, image_size, image_channel, train_path, test_path):
        self.batch_size = batch_size

        self.type_number = type_number
        self.image_size = image_size
        self.image_channel = image_channel

        self._train_images = glob(os.path.join(train_path, "*.jpg"))
        self._test_images = glob(os.path.join(test_path, "*.jpg"))

        self.test_batch_number = len(self._test_images) // self.batch_size
        pass

    def next_train(self):
        begin = np.random.randint(0, len(self._train_images) - self.batch_size)
        return self.norm_image_label(self._train_images[begin: begin + self.batch_size])

    def next_test(self, batch_count):
        begin = self.batch_size * (0 if batch_count >= self.test_batch_number else batch_count)
        return self.norm_image_label(self._test_images[begin: begin + self.batch_size])

    @staticmethod
    def norm_image_label(images_list):
        images = [np.array(misc.imread(image_path).astype(np.float)) / 255.0 for image_path in images_list]
        labels = [int(image_path.split("-")[1].split(".")[0]) for image_path in images_list]
        return images, labels

    pass


class Cifar10Data:

    def __init__(self, batch_size=64, type_number=10, image_size=32,
                 image_channel=3, data_path = "data/cifar-10-batches-py"):
        '''
        
        :param batch_size: 
        :param type_number: 
        :param image_size: 
        :param image_channel: 
        :param data_path: 
        '''

        self._data_path = data_path

        self.batch_size = batch_size

        self.type_number = type_number
        self.image_size = image_size
        self.image_channel = image_channel

        # data
        self.train_images, self.train_labels = self._load(["data_batch_{}".format(i) for i in range(1, 6)])
        self.test_images, self.test_labels = self._load(["test_batch"])

        self.test_batch_number = len(self.test_images) // self.batch_size
        pass

    @staticmethod
    def one_hot(vec, vals=10):
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out

    def _unpickle(self, file):
        with open(os.path.join(self._data_path, file), 'rb') as fo:
            return pickle.load(fo, encoding='latin1')

    def _load(self, source):
        data = [self._unpickle(f) for f in source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255
        # labels = Cifar10Data.one_hot(np.hstack([d["labels"] for d in data]), 10)
        labels = np.hstack([d["labels"] for d in data])
        return images, labels

    def next_test(self, batch_count):
        begin = self.batch_size * (0 if batch_count >= self.test_batch_number else batch_count)
        return self.test_images[begin : begin + self.batch_size], self.test_labels[begin : begin + self.batch_size]

    # 训练阶段，随机选取batch_size个
    def next_train(self):
        ix = np.random.choice(len(self.train_images), self.batch_size)
        return self.train_images[ix], self.train_labels[ix]

    def display_cifar(self, size):
        n = len(self.train_images)
        plt.figure()
        plt.gca().set_axis_off()
        im = np.vstack([np.hstack([self.train_images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
        plt.imshow(im)
        plt.show()

    pass


def create_cifar_image():
    cifar10_data = Cifar10Data()
    print("Number of train images: {}".format(len(cifar10_data.train_images)))
    print("Number of train labels: {}".format(len(cifar10_data.train_labels)))
    print("Number of test images: {}".format(len(cifar10_data.test_images)))
    print("Number of test labels: {}".format(len(cifar10_data.test_labels)))
    cifar10_data.display_cifar(10)


if __name__ == '__main__':

    create_cifar_image()

    pass

