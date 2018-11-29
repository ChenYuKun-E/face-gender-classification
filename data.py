import os
import numpy as np
from PIL import Image
from clipper import Clipper
import uuid

PATH = "/home/face-gender-classification"

IMAGE_W = 28
IMAGE_H = 28


def read_img(files, classify):
    arr = []
    for file in files:
        img = Image.open("%s/%s/%s" % (PATH, classify, file))
        pix = img.load()
        view = np.zeros((IMAGE_H, IMAGE_W, 1), dtype=np.float)
        for x in range(IMAGE_H):
            for y in range(IMAGE_W):
                r, g, b = pix[y, x]
                view[x, y, 0] = (r + g + b) // 3
        arr.append(view)
    return np.array(arr)


def read_label(files):
    arr = []
    for file in files:
        label = int(file[-5:-4])
        view = np.zeros(2, dtype=np.float)
        view[label] = 1
        arr.append(view)
    return np.array(arr)


class Data:

    def __init__(self):
        self.__test_files = os.listdir("%s/test/" % PATH)
        self.__train_files = os.listdir("%s/train/" % PATH)

    def get_train_data(self, limit=0):
        train_files = self.__train_files
        validation_files = []
        if limit > 0 or limit < 0:
            train_files = train_files[:limit]
            validation_files = train_files[limit:]
        return {
            "xs": read_img(train_files, 'train'),
            "labels": read_label(train_files),
            "validation_xs": read_img(validation_files, 'train'),
            "validation_labels": read_label(validation_files),
        }

    def get_test_data(self, limit=0):
        test_files = self.__test_files
        if limit > 0:
            test_files = test_files[:limit]
        return {
            "xs": read_img(test_files, 'test'),
            "names": test_files
        }


if __name__ == '__main__':
    print("init dataset")
    if os.path.exists('%s/train' % PATH):
        print("train dataset exists")
        exit()

    os.mkdir('%s/train' % PATH)
    cli = Clipper()
    for classify in [0, 1]:
        count = 0
        files = os.listdir("%s/%d/" % (PATH, classify))
        for file in files:
            cli.crop_and_save("%s/%d/%s" % (PATH, classify, file),
                              "%s/train/%s-%d.jpg" % (PATH, str(uuid.uuid4()).replace('-', ''), classify))
            count += 1
            print(classify, count)
